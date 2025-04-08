
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

#include "ifrit/runtime/renderer/ayanami/AyanamiDFShadowing.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit.shader/Ayanami/Ayanami.SharedConst.h"

using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Math;

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiDistanceFieldLightingPrivate
    {
        constexpr static u32    kTileAtomicSize    = 64 * 64 * sizeof(u32);
        constexpr static u32    kScatterOutputSize = 64 * 64 * 4096 * sizeof(u32);

        RhiBufferRef            m_TileAtomicsBuf   = nullptr;
        RhiBufferRef            m_ScatterOutputBuf = nullptr;
        RhiTextureRef           m_ScatterOutputTex = nullptr;
        Ref<RhiRenderTargets>   m_RTs              = nullptr;
        Ref<RhiColorAttachment> m_RTColor          = nullptr;

        ResourceNode*           m_ResAtomic           = nullptr;
        ResourceNode*           m_ResScatterOutput    = nullptr;
        ResourceNode*           m_ResScatterOutputTex = nullptr;
    };

    static Matrix4x4f GetLightViewProj(Vector4f sceneBound, Vector3f lightDir)
    {
        Vector3f   normDir  = Normalize(lightDir);
        Vector3f   center   = Vector3f(sceneBound.x, sceneBound.y, sceneBound.z);
        Vector3f   eye      = center - (normDir * sceneBound.w) - 0.01f;
        Vector3f   up       = Vector3f(0.0f, 1.0f, 0.0f);
        Matrix4x4f lookAt   = LookAt(eye, center, up);
        Matrix4x4f proj     = OrthographicNegateY(sceneBound.w * 2, 1.0f, 0.01f, 0.01f + sceneBound.w * 2);
        Matrix4x4f viewProj = MatMul(proj, lookAt);
        return Transpose(viewProj);
    }

    IFRIT_APIDECL GraphicsPassNode& AyanamiDistanceFieldLighting::DistanceFieldShadowTileScatter(
        FrameGraphBuilder& builder, u32 meshDfList, u32 totalMeshDfs, Vector4f sceneBound, Vector3f lightDir,
        u32 tileSize)
    {
        if (tileSize > 64 || totalMeshDfs > 4096)
        {
            iError("Tile size or total mesh distance field exceeds the limit.");
            std::abort();
        }

        // Create the render targets if not already created
        if (!m_Private->m_TileAtomicsBuf.get())
        {
            m_Private->m_TileAtomicsBuf =
                m_Rhi->CreateBuffer("TileAtomicBuffer", AyanamiDistanceFieldLightingPrivate::kTileAtomicSize,
                    RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_SSBO, false, true);
        }
        if (!m_Private->m_ScatterOutputBuf.get())
        {
            m_Private->m_ScatterOutputBuf =
                m_Rhi->CreateBuffer("ScatterOutputBuffer", AyanamiDistanceFieldLightingPrivate::kScatterOutputSize,
                    RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_SSBO, false, true);
        }

        if (!m_Private->m_ScatterOutputTex.get())
        {
            m_Private->m_ScatterOutputTex = m_Rhi->CreateTexture2DMsaa("ScatterOutputTex", tileSize, tileSize,
                RhiImageFormat::RhiImgFmt_B8G8R8A8_SRGB, RhiImageUsage::RhiImgUsage_RenderTarget, 1);
            m_Private->m_RTColor          = m_Rhi->CreateRenderTarget(
                m_Private->m_ScatterOutputTex.get(), { 0, 0, 1, 1 }, RhiRenderTargetLoadOp::Clear, 0, 0);
            m_Private->m_RTs = m_Rhi->CreateRenderTargets();
            m_Private->m_RTs->SetColorAttachments({ m_Private->m_RTColor.get() });
            m_Private->m_RTs->SetRenderArea({ 0, 0, tileSize, tileSize });
        }

        // Prepare the render targets
        struct PushConst
        {
            Matrix4x4f m_VP;
            u32        m_NumMeshDF;
            u32        m_MeshDFDescListId;
            u32        m_NumTilesWidth;
            u32        m_TileAtomics;
            u32        m_ScatterOutput;

        } pc;

        if (sceneBound.w < 1e-4f)
        {
            iError("Scene bound is too small, please check the scene or the light direction.");
            std::abort();
        }

        pc.m_VP               = GetLightViewProj(sceneBound, lightDir);
        pc.m_NumMeshDF        = totalMeshDfs;
        pc.m_MeshDFDescListId = meshDfList;
        pc.m_NumTilesWidth    = 64;
        pc.m_TileAtomics      = m_Private->m_TileAtomicsBuf->GetDescId();
        pc.m_ScatterOutput    = m_Private->m_ScatterOutputBuf->GetDescId();

        m_Private->m_ResAtomic =
            &builder.AddResource("ayainternal_TileAtomic").SetImportedResource(m_Private->m_TileAtomicsBuf.get());
        m_Private->m_ResScatterOutput =
            &builder.AddResource("ayainternal_ScatterOutput").SetImportedResource(m_Private->m_ScatterOutputBuf.get());
        m_Private->m_ResScatterOutputTex =
            &builder.AddResource("ayainternal_ScatterOutputTex")
                 .SetImportedResource(m_Private->m_ScatterOutputTex.get(), { 0, 0, 1, 1 });

        FrameGraphUtils::AddClearUAVPass(builder, "Ayanami.DFShadowCullCleanup", *m_Private->m_ResAtomic, 0);

        FrameGraphUtils::GraphicsPassArgs args;
        args.m_CullMode = Graphics::Rhi::RhiCullMode::Front;

        auto& pass = FrameGraphUtils::AddMeshDrawPass(builder, "Ayanami.DFShadowTileCull",
            Internal::kIntShaderTable.Ayanami.DFShadowTileCullingMS,
            Internal::kIntShaderTable.Ayanami.DFShadowTileCullingFS, m_Private->m_RTs.get(),
            Vector3i{ (i32)totalMeshDfs, 1, 1 }, &pc, FrameGraphUtils::GetPushConstSize<PushConst>(), args);
        auto  pipe = pass.GetPass();

        pass.AddWriteResource(*m_Private->m_ResAtomic)
            .AddReadResource(*m_Private->m_ResScatterOutput)
            .AddReadResource(*m_Private->m_ResScatterOutputTex);

        return pass;
    }

    IFRIT_APIDECL GraphicsPassNode& AyanamiDistanceFieldLighting::DistanceFieldShadowRender(FrameGraphBuilder& builder,
        u32 meshDfList, u32 totalMeshDfs, u32 depthSRV, u32 perframe, Graphics::Rhi::RhiRenderTargets* rts,
        Vector4f sceneBound, Vector3f lightDir, u32 tileSize, float softness)
    {
        using namespace Ifrit::Math;
        struct PushConst
        {
            Matrix4x4f m_LightVP;
            Vector4f   m_LightDir;
            u32        m_TileDFAtomics;
            u32        m_TileDFList;
            u32        m_TotalDFCount;
            u32        m_TileSize;
            u32        m_PerFrameId;
            u32        m_DepthSRV;
            u32        m_MeshDFDescListId;
            f32        m_ShadowCoefK; // This controls DFSS softness.
        } pc;

        auto normDir          = Normalize(lightDir);
        pc.m_LightVP          = GetLightViewProj(sceneBound, lightDir);
        pc.m_LightDir         = Vector4f(normDir.x, normDir.y, normDir.z, 0.0f);
        pc.m_TileDFAtomics    = m_Private->m_TileAtomicsBuf->GetDescId();
        pc.m_TileDFList       = m_Private->m_ScatterOutputBuf->GetDescId();
        pc.m_TotalDFCount     = totalMeshDfs;
        pc.m_TileSize         = tileSize;
        pc.m_PerFrameId       = perframe;
        pc.m_DepthSRV         = depthSRV;
        pc.m_MeshDFDescListId = meshDfList;
        pc.m_ShadowCoefK      = softness;

        auto& pass = FrameGraphUtils::AddPostProcessPass(builder, "Ayanami.DFSS",
            Internal::kIntShaderTable.Ayanami.DFShadowFS, rts, &pc, FrameGraphUtils::GetPushConstSize<PushConst>());
        pass.AddReadResource(*m_Private->m_ResAtomic).AddReadResource(*m_Private->m_ResScatterOutput);
        return pass;
    }

    IFRIT_APIDECL ComputePassNode& AyanamiDistanceFieldLighting::AddDistanceFieldRadianceCachePass(
        FrameGraphBuilder& builder, u32 meshDfList, u32 numTotalMdf, u32 depthAtlasSRV, Vector4f sceneBound,
        Vector3f lightDir, u32 radianceUAV, u32 cardDataId, u32 cardRes, u32 cardAtlasRes, u32 numCards, u32 worldObjId,
        u32 shadowCullTileSize, float softness)
    {
        struct PushConst
        {
            Matrix4x4f m_ShadowLightVP;
            Vector4f   m_ShadowLightDir;

            u32        m_TotalCards;
            u32        m_CardResolution;
            u32        m_CardAtlasResolution;

            u32        m_RadianceUAV;
            u32        m_CardDataId;
            u32        m_DepthAtlasSRVId;

            u32        m_WorldObjId;

            u32        m_ShadowCullTileDFAtomics;
            u32        m_ShadowCullTileDFList;
            u32        m_ShadowCullTileSize;
            u32        m_ShadowCullTotalDFs;
            u32        m_MeshDFDescListId;
            f32        m_ShadowCoefK;
        } pc;

        auto normDir                 = Normalize(lightDir);
        pc.m_ShadowLightVP           = GetLightViewProj(sceneBound, lightDir);
        pc.m_ShadowLightDir          = Vector4f(normDir.x, normDir.y, normDir.z, 0.0f);
        pc.m_TotalCards              = numCards;
        pc.m_CardResolution          = cardRes;
        pc.m_CardAtlasResolution     = cardAtlasRes;
        pc.m_RadianceUAV             = radianceUAV;
        pc.m_CardDataId              = cardDataId;
        pc.m_DepthAtlasSRVId         = depthAtlasSRV;
        pc.m_WorldObjId              = worldObjId;
        pc.m_ShadowCullTileDFAtomics = m_Private->m_TileAtomicsBuf->GetDescId();
        pc.m_ShadowCullTileDFList    = m_Private->m_ScatterOutputBuf->GetDescId();
        pc.m_ShadowCullTileSize      = shadowCullTileSize;
        pc.m_ShadowCullTotalDFs      = numTotalMdf;
        pc.m_MeshDFDescListId        = meshDfList;
        pc.m_ShadowCoefK             = softness;

        auto  cardGroups = DivRoundUp(numCards, Config::kAyanamiRadianceInjectionObjectsPerBlock);
        auto  tileGroups = DivRoundUp(cardRes, Config::kAyanamiRadianceInjectionCardSizePerBlock);

        auto& pass = FrameGraphUtils::AddComputePass(builder, "Ayanami.DFRadianceCachePass",
            Internal::kIntShaderTable.Ayanami.DFRadianceInjectionCS,
            Vector3i{ (i32)tileGroups, (i32)tileGroups, (i32)cardGroups }, &pc, sizeof(PushConst) / sizeof(u32));
        pass.AddReadResource(*m_Private->m_ResAtomic).AddReadResource(*m_Private->m_ResScatterOutput);
        return pass;
    }

    IFRIT_APIDECL AyanamiDistanceFieldLighting::AyanamiDistanceFieldLighting(Graphics::Rhi::RhiBackend* rhi)
        : m_Rhi(rhi), m_Private(new AyanamiDistanceFieldLightingPrivate())
    {
    }

    IFRIT_APIDECL AyanamiDistanceFieldLighting::~AyanamiDistanceFieldLighting() { delete m_Private; }

} // namespace Ifrit::Runtime::Ayanami