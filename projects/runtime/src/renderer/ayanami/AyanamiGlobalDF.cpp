
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

using namespace Ifrit::Math;

namespace Ifrit::Runtime::Ayanami
{

    IFRIT_APIDECL AyanamiGlobalDF::AyanamiGlobalDF(const AyanamiRenderConfig& config, IApplication* app) : m_app(app)
    {
        auto rhi = app->GetRhi();
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
                Graphics::Rhi::RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT
                    | Graphics::Rhi::RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT,
                true);

            m_TestClipMaps[i]->m_clipmapSampler = rhi->CreateTrivialBilinearSampler(false);
            m_TestClipMaps[i]->m_clipmapSRV     = rhi->RegisterCombinedImageSampler(
                m_TestClipMaps[i]->m_clipmapTexture.get(), m_TestClipMaps[i]->m_clipmapSampler.get());
        }
    }

    IFRIT_APIDECL void AyanamiGlobalDF::AddClipmapUpdate(const Graphics::Rhi::RhiCommandList* cmdList, u32 clipmapLevel,
        u32 perFrameDataId, u32 numMeshes, u32 meshDFListId)
    {
        if (m_updateClipmapPass == nullptr)
        {
            m_updateClipmapPass = RenderingUtil::CreateComputePassInternal(
                m_app, Internal::kIntShaderTable.Ayanami.TrivialGlobalDFCompCS, 0, 16);
        }
        auto& clipmap = m_TestClipMaps[clipmapLevel];

        m_updateClipmapPass->SetRecordFunction([&](Graphics::Rhi::RhiRenderPassContext* ctx) {
            auto cmd = ctx->m_cmd;

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

            auto tileRange         = clipmap->m_clipmapSize;
            pc.m_GlobalDFTileRange = Vector4u(tileRange, tileRange, tileRange, 0);
            pc.m_GlobalDFWorldRangeMin =
                Vector4f(clipmap->m_worldBoundMin.x, clipmap->m_worldBoundMin.y, clipmap->m_worldBoundMin.z, 0);
            pc.m_GlobalDFWorldRangeMax =
                Vector4f(clipmap->m_worldBoundMax.x, clipmap->m_worldBoundMax.y, clipmap->m_worldBoundMax.z, 0);
            pc.m_PerFrameDataId   = perFrameDataId;
            pc.m_GlobalDFVolumeId = clipmap->m_clipmapTexture->GetDescId();
            pc.m_NumMeshDF        = numMeshes;
            pc.m_MeshDFDescListId = meshDFListId;

            cmd->SetPushConst(m_updateClipmapPass, 0, sizeof(PushConst), &pc);
            auto tgX = DivRoundUp(tileRange, Config::kAyanamiGlobalDFCompositeTileSize);
            cmd->Dispatch(tgX, tgX, tgX);
        });

        cmdList->BeginScope("Ayanami: GlobalDFComposite");
        m_updateClipmapPass->Run(cmdList, 0);
        cmdList->EndScope();
    }

    IFRIT_APIDECL void AyanamiGlobalDF::AddRayMarchPass(const Graphics::Rhi::RhiCommandList* cmdList, u32 clipmapLevel,
        u32 perFrameDataId, u32 outTextureId, Vector2u outTextureSize)
    {
        if (m_raymarchPass == nullptr)
        {
            m_raymarchPass = RenderingUtil::CreateComputePassInternal(
                m_app, Internal::kIntShaderTable.Ayanami.GlobalDFRayMarchCS, 0, 17);
        }

        auto& clipmap = m_TestClipMaps[clipmapLevel];
        m_raymarchPass->SetRecordFunction([&](Graphics::Rhi::RhiRenderPassContext* ctx) {
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

            auto cmd = ctx->m_cmd;
            auto tgX = DivRoundUp(outTextureSize.x, Config::kAyanamiGlobalDFRayMarchTileSize);
            auto tgY = DivRoundUp(outTextureSize.y, Config::kAyanamiGlobalDFRayMarchTileSize);

            cmd->SetPushConst(m_raymarchPass, 0, sizeof(RayMarchPc), &pc);
            cmd->Dispatch(tgX, tgY, 1);
        });

        cmdList->BeginScope("Ayanami: GlobalDFRayMarch");
        m_raymarchPass->Run(cmdList, 0);
        cmdList->EndScope();
    }

    IFRIT_APIDECL AyanamiGlobalDF::GPUTexture AyanamiGlobalDF::GetClipmapVolume(u32 clipmapLevel)
    {
        return m_TestClipMaps[clipmapLevel]->m_clipmapTexture;
    }

} // namespace Ifrit::Runtime::Ayanami