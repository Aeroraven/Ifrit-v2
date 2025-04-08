
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/runtime/renderer/ayanami/AyanamiGlobalDF.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"
#include "ifrit/core/math/constfunc/ConstFunc.h"

#include "ifrit.shader/Ayanami/Ayanami.SharedConst.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"

using namespace Ifrit::Math;

namespace Ifrit::Runtime::Ayanami
{

    IFRIT_APIDECL AyanamiGlobalDF::AyanamiGlobalDF(const AyanamiRenderConfig& config, IApplication* app) : m_app(app)
    {
        auto rhi                = app->GetRhi();
        auto linearClampSampler = app->GetSharedRenderResource()->GetLinearClampSampler();
        m_TestClipMaps.resize(config.m_globalDFClipmapLevels);
        for (u32 i = 0; i < config.m_globalDFClipmapLevels; i++)
        {
            auto extent                        = config.m_globalDFBaseExtent;
            auto resolution                    = config.m_globalDFClipmapResolution;
            m_TestClipMaps[i]                  = std::make_unique<AyanamiGlobalDFClipmap>();
            m_TestClipMaps[i]->m_clipmapSize   = resolution;
            m_TestClipMaps[i]->m_worldBoundMin = Vector3f(-extent, -extent, -extent);
            m_TestClipMaps[i]->m_worldBoundMax = Vector3f(extent, extent, extent);

            m_TestClipMaps[i]->m_clipmapTexture = rhi->CreateTexture3D("Ayanami_GlobalDF", resolution, resolution,
                resolution, Graphics::Rhi::RhiImageFormat::RhiImgFmt_R32_SFLOAT,
                Graphics::Rhi::RhiImageUsage::RhiImgUsage_UnorderedAccess
                    | Graphics::Rhi::RhiImageUsage::RhiImgUsage_ShaderRead,
                true);

            m_TestClipMaps[i]->m_clipmapSRV =
                rhi->RegisterCombinedImageSampler(m_TestClipMaps[i]->m_clipmapTexture.get(), linearClampSampler.get());

            // Voxel Lighting Resources
            u32 totalVoxels = config.m_VoxelExtentPerGlobalClipMap * config.m_VoxelExtentPerGlobalClipMap
                * config.m_VoxelExtentPerGlobalClipMap;
            m_TestClipMaps[i]->m_VoxelsPerWidth   = config.m_VoxelExtentPerGlobalClipMap;
            m_TestClipMaps[i]->m_objectGridBuffer = rhi->CreateBuffer("Ayanami_GlobalDF_ObjectGrid",
                totalVoxels * sizeof(u32) * Config::kAyanami_MaxObjectPerGridCell,
                Graphics::Rhi::RhiBufferUsage::RhiBufferUsage_SSBO, false, true);
        }
    }

    IFRIT_APIDECL ComputePassNode& AyanamiGlobalDF::AddClipmapUpdate(
        FrameGraphBuilder& builder, u32 clipmapLevel, u32 perFrameDataId, u32 numMeshes, u32 meshDFListId)
    {
        auto& clipmap   = m_TestClipMaps[clipmapLevel];
        auto  tileRange = clipmap->m_clipmapSize;
        auto  tgX       = DivRoundUp(tileRange, Config::kAyanamiGlobalDFCompositeTileSize);
        struct PushConst
        {
            Vector4u m_GlobalDFTileRange;
            Vector4f m_GlobalDFWorldRangeMin;
            Vector4f m_GlobalDFWorldRangeMax;
            u32      m_PerFrameDataId;
            u32      m_GlobalDFVolumeId;
            u32      m_NumMeshDF;
            u32      m_MeshDFDescListId;
        } pc;
        pc.m_GlobalDFTileRange = Vector4u(tileRange, tileRange, tileRange, 0);
        pc.m_GlobalDFWorldRangeMin =
            Vector4f(clipmap->m_worldBoundMin.x, clipmap->m_worldBoundMin.y, clipmap->m_worldBoundMin.z, 0);
        pc.m_GlobalDFWorldRangeMax =
            Vector4f(clipmap->m_worldBoundMax.x, clipmap->m_worldBoundMax.y, clipmap->m_worldBoundMax.z, 0);
        pc.m_PerFrameDataId   = perFrameDataId;
        pc.m_GlobalDFVolumeId = clipmap->m_clipmapTexture->GetDescId();
        pc.m_NumMeshDF        = numMeshes;
        pc.m_MeshDFDescListId = meshDFListId;

        auto& pass = FrameGraphUtils::AddComputePass(builder, "Ayanami.GlobalDFComposite",
            Internal::kIntShaderTable.Ayanami.TrivialGlobalDFCompCS, Vector3i{ (i32)tgX, (i32)tgX, 1 }, &pc,
            sizeof(PushConst) / sizeof(u32));
        return pass;
    }

    IFRIT_APIDECL ComputePassNode& AyanamiGlobalDF::AddRayMarchPass(
        FrameGraphBuilder& builder, u32 clipmapLevel, u32 perFrameDataId, u32 outTextureId, Vector2u outTextureSize)
    {
        auto& clipmap = m_TestClipMaps[clipmapLevel];
        struct RayMarchPc
        {
            Vector4f m_GlobalDFBoxMin;
            Vector4f m_GlobalDFBoxMax;
            u32      m_PerFrameId;
            u32      m_GlobalDFId;
            u32      m_OutTex;
            u32      m_RtH;
            u32      m_RtW;
        } pc;
        pc.m_GlobalDFBoxMax =
            Vector4f(clipmap->m_worldBoundMax.x, clipmap->m_worldBoundMax.y, clipmap->m_worldBoundMax.z, 0);
        pc.m_GlobalDFBoxMin =
            Vector4f(clipmap->m_worldBoundMin.x, clipmap->m_worldBoundMin.y, clipmap->m_worldBoundMin.z, 0);
        pc.m_PerFrameId = perFrameDataId;
        pc.m_GlobalDFId = clipmap->m_clipmapSRV->GetActiveId();

        pc.m_OutTex = outTextureId;
        pc.m_RtH    = outTextureSize.y;
        pc.m_RtW    = outTextureSize.x;

        auto& pass = FrameGraphUtils::AddComputePass(builder, "Ayanami.GlobalDFRayMarch",
            Internal::kIntShaderTable.Ayanami.GlobalDFRayMarchCS,
            Vector3i{ DivRoundUp<i32, i32>(outTextureSize.x, Config::kAyanamiGlobalDFRayMarchTileSize),
                DivRoundUp<i32, i32>(outTextureSize.x, Config::kAyanamiGlobalDFRayMarchTileSize), 1 },
            &pc, sizeof(RayMarchPc) / sizeof(u32));
        return pass;
    }

    IFRIT_APIDECL ComputePassNode& AyanamiGlobalDF::AddObjectGridCompositionPass(
        FrameGraphBuilder& builder, u32 clipmapLevel, u32 numMeshes, u32 meshDFListId)
    {
        struct PushConst
        {
            u32 m_NumTotalMeshDF;
            u32 m_MeshDFDescListId;
            f32 m_ClipMapRadius;
            u32 m_VoxelsPerClipMapWidth;
            u32 m_CellDataId;
        } pc;

        pc.m_NumTotalMeshDF        = numMeshes;
        pc.m_MeshDFDescListId      = meshDFListId;
        pc.m_ClipMapRadius         = m_TestClipMaps[clipmapLevel]->m_worldBoundMax.z;
        pc.m_VoxelsPerClipMapWidth = m_TestClipMaps[clipmapLevel]->m_VoxelsPerWidth;
        pc.m_CellDataId            = m_TestClipMaps[clipmapLevel]->m_objectGridBuffer->GetDescId();

        u32 groupsX =
            DivRoundUp<u32, u32>(m_TestClipMaps[clipmapLevel]->m_VoxelsPerWidth, Config::kAyanamiObjectGridTileSize);
        auto& pass = FrameGraphUtils::AddComputePass(builder, "Ayanami.ObjectGridComposition",
            Internal::kIntShaderTable.Ayanami.ObjectGridCompositionCS,
            Vector3i{ (int)groupsX, (int)groupsX, (int)groupsX }, &pc, FrameGraphUtils::GetPushConstSize<PushConst>());
        return pass;
    }

    IFRIT_APIDECL AyanamiGlobalDF::GPUTexture AyanamiGlobalDF::GetClipmapVolume(u32 clipmapLevel)
    {
        return m_TestClipMaps[clipmapLevel]->m_clipmapTexture;
    }

    IFRIT_APIDECL u32 AyanamiGlobalDF::GetClipmapVolumeSRV(u32 clipmapLevel)
    {
        return m_TestClipMaps[clipmapLevel]->m_clipmapSRV->GetActiveId();
    }

} // namespace Ifrit::Runtime::Ayanami