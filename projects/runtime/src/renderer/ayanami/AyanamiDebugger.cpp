
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

#include "ifrit/runtime/renderer/ayanami/AyanamiDebugger.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Ayanami.h"
#include "ifrit.shader/Ayanami/Ayanami.SharedConst.h"

using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime::FrameGraphUtils;

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiDebuggerPrivate
    {
        RhiTextureRef m_RenderAtomicDepthAtlas = nullptr;
    };

    IFRIT_APIDECL AyanamiDebugger::AyanamiDebugger(Graphics::Rhi::RhiBackend* rhi) : m_Rhi(rhi)
    {
        m_Private = new AyanamiDebuggerPrivate();
    }
    IFRIT_APIDECL                  AyanamiDebugger::~AyanamiDebugger() { delete m_Private; }

    IFRIT_APIDECL ComputePassNode& AyanamiDebugger::RenderSceneFromCacheSurface(FrameGraphBuilder& builder,
        FGTextureNodeRef outputTexture, FGTextureNodeRef cardAlbedoAtlas, FGTextureNodeRef cardNormalAtlas,
        FGTextureNodeRef cardRadianceAtlas, FGTextureNodeRef cardDepthAtlas, u32 totalCards, u32 cardResolution,
        u32 cardAtlasResolution, u32 cardDataBuffer, u32 perFrameId, u32 meshDfListId)
    {
        auto  rtWidth        = outputTexture->GetWidth();
        auto  rtHeight       = outputTexture->GetHeight();
        auto& resAtomicDepth = builder.DeclareTexture("Ayanami.Debug.RenderAtomicDepthAtlas",
            FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImgFmt_R64_UINT,
                RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_CopyDst));

        AddClearUAVTexturePass(
            builder, "Ayanami.Debug.ReconFromSurfaceCache.DepthClear", resAtomicDepth, 0xffffffffffffffffull);

        struct PushConst
        {
            u32 m_NumTotalCards;
            u32 m_CardResolution;
            u32 m_CardAtlasResolution;
            u32 m_PerFrameDataId;
            u32 m_CardDataBufferId;
            u32 m_OutputUAV;
            u32 m_CardAlbedoAtlasSRV;
            u32 m_CardNormalAtlasSRV;
            u32 m_CardRadianceAtlasSRV;
            u32 m_CardDepthAtlasSRV;
            u32 m_AtomicDepthUAV;
            u32 m_OutputWidth;
            u32 m_OutputHeight;
            u32 m_MeshDFDescId;
        } pc;

        pc.m_NumTotalCards       = totalCards;
        pc.m_CardResolution      = cardResolution;
        pc.m_CardAtlasResolution = cardAtlasResolution;
        pc.m_PerFrameDataId      = perFrameId;
        pc.m_CardDataBufferId    = cardDataBuffer;
        pc.m_OutputWidth         = rtWidth;
        pc.m_OutputHeight        = rtHeight;
        pc.m_MeshDFDescId        = meshDfListId;

        pc.m_OutputUAV            = 0;
        pc.m_CardAlbedoAtlasSRV   = 0;
        pc.m_CardNormalAtlasSRV   = 0;
        pc.m_CardRadianceAtlasSRV = 0;
        pc.m_CardDepthAtlasSRV    = 0;
        pc.m_AtomicDepthUAV       = 0;

        auto  tgX       = DivRoundUp(cardResolution, Config::kAyanamiReconFromSCTileSize);
        auto  atomicPtr = &resAtomicDepth;
        auto& pass1     = AddComputePass<PushConst>(builder, "Ayanami.Debug.DepthReconFromSurfaceCache",
            Internal::kIntShaderTableAyanami.DbgReconFromSurfaceCacheCS, Vector3i((i32)tgX, (i32)tgX, (i32)totalCards),
            pc,
            [outputTexture, cardAlbedoAtlas, cardNormalAtlas, cardRadianceAtlas, cardDepthAtlas, atomicPtr](
                PushConst data, const FrameGraphPassContext& ctx) {
                data.m_OutputUAV            = ctx.m_FgDesc->GetUAV(*outputTexture);
                data.m_CardAlbedoAtlasSRV   = ctx.m_FgDesc->GetSRV(*cardAlbedoAtlas);
                data.m_CardNormalAtlasSRV   = ctx.m_FgDesc->GetSRV(*cardNormalAtlas);
                data.m_CardRadianceAtlasSRV = ctx.m_FgDesc->GetSRV(*cardRadianceAtlas);
                data.m_CardDepthAtlasSRV    = ctx.m_FgDesc->GetSRV(*cardDepthAtlas);
                data.m_AtomicDepthUAV       = ctx.m_FgDesc->GetUAV(*atomicPtr);
                SetRootSignature(data, ctx);
            });
        pass1.AddReadResource(*cardAlbedoAtlas)
            .AddWriteResource(*outputTexture)
            .AddReadResource(*cardNormalAtlas)
            .AddReadResource(*cardRadianceAtlas)
            .AddReadResource(*cardDepthAtlas)
            .AddReadWriteResource(resAtomicDepth);

        auto  tgX2  = DivRoundUp(rtWidth, Config::kAyanamiReconFromSCDepthTileSize);
        auto  tgY2  = DivRoundUp(rtHeight, Config::kAyanamiReconFromSCDepthTileSize);
        auto& pass2 = AddComputePass<PushConst>(builder, "Ayanami.Debug.SampleReconDepth",
            Internal::kIntShaderTableAyanami.DbgSampleReconDepthCS, Vector3i((i32)tgX2, (i32)tgY2, 1), pc,
            [outputTexture, cardAlbedoAtlas, cardNormalAtlas, cardRadianceAtlas, cardDepthAtlas, atomicPtr](
                PushConst data, const FrameGraphPassContext& ctx) {
                data.m_OutputUAV            = ctx.m_FgDesc->GetUAV(*outputTexture);
                data.m_CardAlbedoAtlasSRV   = ctx.m_FgDesc->GetSRV(*cardAlbedoAtlas);
                data.m_CardNormalAtlasSRV   = ctx.m_FgDesc->GetSRV(*cardNormalAtlas);
                data.m_CardRadianceAtlasSRV = ctx.m_FgDesc->GetSRV(*cardRadianceAtlas);
                data.m_CardDepthAtlasSRV    = ctx.m_FgDesc->GetSRV(*cardDepthAtlas);
                data.m_AtomicDepthUAV       = ctx.m_FgDesc->GetUAV(*atomicPtr);
                SetRootSignature(data, ctx);
            });
        pass2.AddReadResource(*cardDepthAtlas).AddReadResource(resAtomicDepth).AddWriteResource(*outputTexture);

        return pass1;
    }
} // namespace Ifrit::Runtime::Ayanami