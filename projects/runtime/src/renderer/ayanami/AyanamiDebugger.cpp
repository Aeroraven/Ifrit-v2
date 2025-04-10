
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
    IFRIT_APIDECL      AyanamiDebugger::~AyanamiDebugger() { delete m_Private; }

    IFRIT_APIDECL void AyanamiDebugger::RenderSceneFromCacheSurface(FrameGraphBuilder& builder,
        FGTextureNodeRef outputTexture, FGTextureNodeRef cardAlbedoAtlas, FGTextureNodeRef cardNormalAtlas,
        FGTextureNodeRef cardRadianceAtlas, FGTextureNodeRef cardDepthAtlas, u32 totalCards, u32 cardResolution,
        u32 cardAtlasResolution, u32 cardDataBuffer, u32 perFrameId, u32 meshDfListId)
    {
        auto  rtWidth        = outputTexture->GetWidth();
        auto  rtHeight       = outputTexture->GetHeight();
        auto& resAtomicDepth = builder.DeclareTexture("Ayanami.Debug.RenderAtomicDepthAtlas",
            FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImgFmt_R64_UINT,
                RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_CopyDst));

        // Note: RenderDoc might show INCORRECT value on R64_UINT clear.
        // following behavior is well-defined in vulkan spec.
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
    }

    void AyanamiDebugger::RenderSceneFromSamplingObjectGrids(FrameGraphBuilder& builder, FGTextureNodeRef outputTexture,
        FGTextureNodeRef cardDirectLightingAtlas, FGTextureNodeRef cardAlbedoAtlas, FGTextureNodeRef cardDepthAtlas,
        FGTextureNodeRef globalDF, FGBufferNodeRef globalObjectGrids, u32 perFrameDataId, u32 allCardData,
        u32 meshDfDesc, Vector2u outTextureSize, Vector3f worldBoundMax, Vector3f worldBoundMin, u32 cardResolution,
        u32 cardAtlasResolution, u32 voxelsPerWidth, u32 globalDfWidth)
    {
        struct PushConst
        {
            Vector4f m_GlobalDFBoxMin;
            Vector4f m_GlobalDFBoxMax;
            u32      m_PerFrameId;
            u32      m_GlobalDFId;
            u32      m_OutTex;
            u32      m_RtH;
            u32      m_RtW;
            u32      m_GlobalObjectGridUAV;
            u32      m_GlobalDFResolution;
            u32      m_VoxelsPerClipMapWidth;
            u32      m_MeshDFDescListId;
            u32      m_AllCardData;
            u32      m_CardResolution;
            u32      m_CardAtlasResolution;
            u32      m_CardDepthAtlasSRV;
            u32      m_CardAlbedoAtlasSRV;
        } pc;
        pc.m_GlobalDFBoxMax        = Vector4f(worldBoundMax, 0.0f);
        pc.m_GlobalDFBoxMin        = Vector4f(worldBoundMin, 0.0f);
        pc.m_PerFrameId            = perFrameDataId;
        pc.m_GlobalDFId            = 0;
        pc.m_OutTex                = 0;
        pc.m_RtH                   = outTextureSize.y;
        pc.m_RtW                   = outTextureSize.x;
        pc.m_GlobalObjectGridUAV   = 0;
        pc.m_GlobalDFResolution    = globalDfWidth;
        pc.m_VoxelsPerClipMapWidth = voxelsPerWidth;
        pc.m_MeshDFDescListId      = meshDfDesc;
        pc.m_AllCardData           = allCardData;
        pc.m_CardResolution        = cardResolution;      // TODO
        pc.m_CardAtlasResolution   = cardAtlasResolution; // TODO
        pc.m_CardDepthAtlasSRV     = 0;
        pc.m_CardAlbedoAtlasSRV    = 0;

        auto tgX = DivRoundUp(outTextureSize.x, Config::kAyanamiDbgObjGridTileSize);
        auto tgY = DivRoundUp(outTextureSize.y, Config::kAyanamiDbgObjGridTileSize);

        AddComputePass<PushConst>(builder, "Ayanami.Debug.SampleObjectGrids",
            Internal::kIntShaderTableAyanami.DbgSampleObjectGridsCS, Vector3i{ (i32)tgX, (i32)tgY, 1 }, pc,
            [outputTexture, globalDF, globalObjectGrids, cardDepthAtlas, cardAlbedoAtlas](
                PushConst data, const FrameGraphPassContext& ctx) {
                data.m_OutTex              = ctx.m_FgDesc->GetUAV(*outputTexture);
                data.m_GlobalDFId          = ctx.m_FgDesc->GetSRV(*globalDF);
                data.m_GlobalObjectGridUAV = ctx.m_FgDesc->GetUAV(*globalObjectGrids);
                data.m_CardDepthAtlasSRV   = ctx.m_FgDesc->GetSRV(*cardDepthAtlas);
                data.m_CardAlbedoAtlasSRV  = ctx.m_FgDesc->GetSRV(*cardAlbedoAtlas);
                SetRootSignature(data, ctx);
            })
            .AddWriteResource(*outputTexture)
            .AddReadResource(*cardDirectLightingAtlas)
            .AddReadResource(*cardAlbedoAtlas)
            .AddReadResource(*globalObjectGrids)
            .AddReadResource(*cardDepthAtlas)
            .AddReadResource(*globalDF);
    }

    IFRIT_APIDECL void AyanamiDebugger::RenderValidObjectGrids(FrameGraphBuilder& builder,
        FGTextureNodeRef outputTexture, FGBufferNodeRef globalObjectGrids, Vector3f worldBoundMax,
        Vector3f worldBoundMin, u32 voxelsPerWidth, u32 perFrame)
    {
        auto  rtWidth  = outputTexture->GetWidth();
        auto  rtHeight = outputTexture->GetHeight();

        auto& resTempDepth = builder.DeclareTexture("Ayanami.RDG.DebugValidObjectGrids_Depth",
            FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImgFmt_D32_SFLOAT, RhiImageUsage::RhiImgUsage_Depth));

        struct PushConst
        {
            Vector4f m_WorldBoundMin;
            Vector4f m_WorldBoundMax;
            u32      m_PerFrame;
            u32      m_VoxelsPerWidth;
            u32      m_ObjectGridId;
        } pc;

        pc.m_WorldBoundMin  = Vector4f(worldBoundMin, 0.0f);
        pc.m_WorldBoundMax  = Vector4f(worldBoundMax, 0.0f);
        pc.m_PerFrame       = perFrame;
        pc.m_VoxelsPerWidth = voxelsPerWidth;
        pc.m_ObjectGridId   = 0;

        i32              tgX = voxelsPerWidth;
        GraphicsPassArgs args;
        args.m_CullMode = Graphics::Rhi::RhiCullMode::None;

        AddMeshDrawPass<PushConst>(builder, "Ayanami.Debug.ValidObjectGrids",
            Internal::kIntShaderTableAyanami.DbgVisObjGridsMS, Internal::kIntShaderTableAyanami.DbgVisObjGridsFS,
            Vector3i(tgX), args, pc,
            [globalObjectGrids](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_ObjectGridId = ctx.m_FgDesc->GetSRV(*globalObjectGrids);
                SetRootSignature(data, ctx);
            })
            .AddRenderTarget(*outputTexture)
            .AddDepthTarget(resTempDepth)
            .AddReadResource(*globalObjectGrids);
    }
} // namespace Ifrit::Runtime::Ayanami