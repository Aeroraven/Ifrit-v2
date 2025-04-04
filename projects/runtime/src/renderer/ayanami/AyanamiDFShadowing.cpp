
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
    };

    IFRIT_APIDECL GraphicsPassNode& AyanamiDistanceFieldLighting::DistanceFieldShadowTileScatter(
        FrameGraphBuilder& builder, u32 meshDfList, u32 totalMeshDfs, Vector4f sceneBound, Vector3f lightDir,
        u32 tileSize)
    {
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
            m_Private->m_ScatterOutputTex = m_Rhi->CreateTexture2D("ScatterOutputTex", tileSize, tileSize,
                RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT, false);
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

        Vector3f   normDir  = Normalize(lightDir);
        Vector3f   center   = Vector3f(sceneBound.x, sceneBound.y, sceneBound.z);
        Vector3f   eye      = center - (normDir * sceneBound.w) - 0.01f;
        Vector3f   up       = Vector3f(0.0f, 1.0f, 0.0f);
        Matrix4x4f lookAt   = LookAt(eye, center, up);
        Matrix4x4f proj     = OrthographicNegateY(sceneBound.w * 2, 1.0f, 0.01f, 0.01f + sceneBound.w * 2);
        Matrix4x4f viewProj = MatMul(lookAt, proj);

        pc.m_VP               = viewProj;
        pc.m_NumMeshDF        = totalMeshDfs;
        pc.m_MeshDFDescListId = meshDfList;
        pc.m_NumTilesWidth    = 64;
        pc.m_TileAtomics      = m_Private->m_TileAtomicsBuf->GetDescId();
        pc.m_ScatterOutput    = m_Private->m_ScatterOutputBuf->GetDescId();

        auto& resAtomic =
            builder.AddResource("ayainternal_TileAtomic").SetImportedResource(m_Private->m_TileAtomicsBuf.get());
        auto& resScatterOutput =
            builder.AddResource("ayainternal_ScatterOutput").SetImportedResource(m_Private->m_ScatterOutputBuf.get());
        auto& resScatterOutputTex = builder.AddResource("ayainternal_ScatterOutputTex")
                                        .SetImportedResource(m_Private->m_ScatterOutputTex.get(), { 0, 0, 1, 1 });

        FrameGraphUtils::AddClearUAVPass(builder, "Ayanami/DFShadowCullCleanup", resAtomic, 0);

        auto& pass = FrameGraphUtils::AddMeshDrawPass(builder, "Ayanami/DFShadowTileCull",
            Internal::kIntShaderTable.Ayanami.DFShadowTileCullingMS,
            Internal::kIntShaderTable.Ayanami.DFShadowTileCullingFS, m_Private->m_RTs.get(),
            Vector3i{ (i32)totalMeshDfs, 1, 1 }, &pc, sizeof(PushConst) / sizeof(u32));
        pass.AddWriteResource(resAtomic).AddReadResource(resScatterOutputTex).AddReadResource(resScatterOutput);

        return pass;
    }

    IFRIT_APIDECL AyanamiDistanceFieldLighting::AyanamiDistanceFieldLighting(Graphics::Rhi::RhiBackend* rhi)
        : m_Rhi(rhi), m_Private(new AyanamiDistanceFieldLightingPrivate())
    {
    }

    IFRIT_APIDECL AyanamiDistanceFieldLighting::~AyanamiDistanceFieldLighting() { delete m_Private; }

} // namespace Ifrit::Runtime::Ayanami