
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

#include "ifrit/runtime/renderer/ayanami/AyanamiScreenProbeProc.h"
#include "ifrit.shader/Ayanami/Ayanami.SharedConst.h"
#include "ifrit/runtime/rendercore/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Ayanami.h"
#include "ifrit/core/math/sampling/LowDiscrepancy.h"

using namespace Ifrit::RHI;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime::FrameGraphUtils;
using namespace Ifrit::Runtime::Ayanami::Config;

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiScreenProbeProcessorPrivate
    {
        f32               m_AdaptiveProbesRatio      = 0.0f;
        u32               m_MaxRTWidth               = 0;
        u32               m_MaxRTHeight              = 0;
        u32               m_MaxUniformTilesPerWidth  = 0;
        u32               m_MaxUniformTilesPerHeight = 0;
        u32               m_MaxUniformProbes         = 0;
        u32               m_MaxAdaptiveProbesCount   = 0;

        FGTextureNodeRef  m_RadianceAtlas         = nullptr;
        FGTextureNodeRef  m_RadianceAtlasFixed    = nullptr;
        FGBufferNodeRef   m_AdaptiveProbesList    = nullptr;
        FGBufferNodeRef   m_AdaptiveProbesCounter = nullptr;

        FGBufferNodeRef   m_MeshDFTracingRayList         = nullptr;
        FGBufferNodeRef   m_MeshDFTracingRayIndirectArgs = nullptr;
        FGBufferNodeRef   m_CubeIndex                    = nullptr;
        FGTextureNodeRef  m_MeshDFCullingDepth           = nullptr;
        FGTextureNodeRef  m_MeshDFCullingDummy           = nullptr;
        FGBufferNodeRef   m_MeshDFCullingMatrix          = nullptr;
        FGBufferNodeRef   m_MeshDFCullingListCounter     = nullptr;
        FGBufferNodeRef   m_MeshDFCullingList            = nullptr;
        FGBufferNodeRef   m_MeshDFCullingIndirectArgs    = nullptr;

        FGBufferNodeRef   m_GlobalDFTracingList         = nullptr;
        FGBufferNodeRef   m_GlobalDFTracingIndirectArgs = nullptr;

        FGBufferNodeRef   m_IntegratedSH = nullptr;

        FGTextureNodeRef  m_ActiveGBufferAlbedo = nullptr;

        u32               m_ActiveRTWidth  = 0;
        u32               m_ActiveRTHeight = 0;
        Vector4f          m_ActiveWorldBoundMin;
        Vector4f          m_ActiveWorldBoundMax;
        u32               m_ActiveMDFCounts = 0;

        u32               m_MDFCullGridSizeXY   = 16;
        u32               m_MDFCullGridSizeZ    = 1;
        u32               m_MDFMaxCullObjInGrid = 512;

        // Persistent Resources
        RHI::RhiBufferRef m_CubeIndexRHI = nullptr;
        Vector2f          m_RayJitter    = Vector2f(0.0f, 0.0f);
    };

    IFRIT_APIDECL AyanamiScreenProbeProcessor::AyanamiScreenProbeProcessor(
        RHI::RhiBackend* rhi, AyanamiSharedContext* sharedContext)
        : m_Rhi(rhi), m_SharedContext(sharedContext)
    {
        m_Private = new AyanamiScreenProbeProcessorPrivate();
    }

    IFRIT_APIDECL      AyanamiScreenProbeProcessor::~AyanamiScreenProbeProcessor() { delete m_Private; }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::InitContext(
        FrameGraphBuilder& builder, u32 maxRtWidth, u32 maxRtHeight, f32 adaptiveProbesRatio)
    {
        m_Private->m_AdaptiveProbesRatio = adaptiveProbesRatio;
        m_Private->m_MaxRTWidth          = maxRtWidth;
        m_Private->m_MaxRTHeight         = maxRtHeight;

        m_Private->m_MaxUniformTilesPerHeight = DivRoundUp(maxRtHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
        m_Private->m_MaxUniformTilesPerWidth  = DivRoundUp(maxRtWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
        m_Private->m_MaxUniformProbes = m_Private->m_MaxUniformTilesPerHeight * m_Private->m_MaxUniformTilesPerWidth;

        m_Private->m_MaxAdaptiveProbesCount = static_cast<u32>(m_Private->m_MaxUniformProbes * adaptiveProbesRatio);

        // Then, create a adaptive probe list. each element stores a packed uint32 value for screen offset
        m_Private->m_AdaptiveProbesList = &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.AdaptiveProbesList",
            FrameGraphBufferDesc(
                m_Private->m_MaxAdaptiveProbesCount * sizeof(u32), RhiBufferUsage::RhiBufferUsage_SSBO));

        m_Private->m_AdaptiveProbesCounter = &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.AdaptiveProbesCounter",
            FrameGraphBufferDesc(sizeof(u32) * 10,
                RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst
                    | RhiBufferUsage::RhiBufferUsage_Indirect));

        // m_MeshDFTracingRayIndirectArgs: (counter | indargs for mesh df tracing)
        m_Private->m_MeshDFTracingRayIndirectArgs =
            &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.MeshDFTracingIndirectArgs",
                FrameGraphBufferDesc(sizeof(u32) * 4,
                    RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst
                        | RhiBufferUsage::RhiBufferUsage_Indirect));

        m_Private->m_GlobalDFTracingIndirectArgs =
            &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.GlobalDFTracingIndirectArgs",
                FrameGraphBufferDesc(sizeof(u32) * 4,
                    RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst
                        | RhiBufferUsage::RhiBufferUsage_Indirect));

        // Create the radiance atlas
        {
            u32 requiredHeightBase = m_Private->m_MaxUniformTilesPerWidth * kAyanami_ScreenProbeProbeHemiRes;
            u32 requiredWidthBase  = m_Private->m_MaxUniformTilesPerHeight * kAyanami_ScreenProbeProbeHemiRes;
            u32 requiredHeightAdaptive =
                static_cast<u32>(std::ceil(m_Private->m_AdaptiveProbesRatio * requiredHeightBase));
            u32 requiredHeight = requiredHeightBase + requiredHeightAdaptive;
            u32 requiredWidth  = requiredWidthBase;

            m_Private->m_RadianceAtlas = &builder.DeclareTexture("Ayanami.RDG.ScreeProbe.RadianceAtlas",
                FrameGraphTextureDesc(requiredWidth, requiredHeight, 1, RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_ShaderRead));
        }

        {
            u32 requiredHeightBase = m_Private->m_MaxUniformTilesPerWidth * (kAyanami_ScreenProbeProbeHemiRes + 2);
            u32 requiredWidthBase  = m_Private->m_MaxUniformTilesPerHeight * (kAyanami_ScreenProbeProbeHemiRes + 2);

            u32 requiredHeightAdaptive =
                static_cast<u32>(std::ceil(m_Private->m_AdaptiveProbesRatio * requiredHeightBase));
            u32 requiredHeight = requiredHeightBase + requiredHeightAdaptive;
            u32 requiredWidth  = requiredWidthBase;

            m_Private->m_RadianceAtlasFixed = &builder.DeclareTexture("Ayanami.RDG.ScreeProbe.RadianceAtlasFixed",
                FrameGraphTextureDesc(requiredWidth, requiredHeight, 1, RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_ShaderRead));
        }

        // Then list for trace coords for mesh df tracing
        {
            u32 maxProbes                     = m_Private->m_MaxUniformProbes + m_Private->m_MaxAdaptiveProbesCount;
            u32 maxTraces                     = maxProbes * kAyanami_ScreenProbeTracePerProbe;
            m_Private->m_MeshDFTracingRayList = &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.MeshDFTracingRayList",
                FrameGraphBufferDesc(maxTraces * sizeof(u32), RhiBufferUsage::RhiBufferUsage_SSBO));
            m_Private->m_GlobalDFTracingList  = &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.GlobalDFTracingList",
                 FrameGraphBufferDesc(maxTraces * sizeof(u32), RhiBufferUsage::RhiBufferUsage_SSBO));
        }

        // then probes SH
        {
            u32 maxProbes = m_Private->m_MaxUniformProbes + m_Private->m_MaxAdaptiveProbesCount;
            u32 maxSHSize = maxProbes * 27;

            m_Private->m_IntegratedSH = &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.IntegratedSH",
                FrameGraphBufferDesc(maxSHSize * sizeof(Vector4f), RhiBufferUsage::RhiBufferUsage_SSBO));
        }

        // Prepare the cube index buffer
        {
            if (m_Private->m_CubeIndexRHI == nullptr)
            {

                // a clumsy workaround to make clang-format happy
#define TMP_CUBE_INDICES \
    0, 1, 2, 1, 3, 2, 5, 4, 7, 4, 6, 7, 4, 5, 0, 5, 1, 0, 4, 0, 6, 0, 2, 6, 1, 5, 3, 5, 7, 3, 2, 3, 6, 3, 7, 6,
                Array<u32, 36> cubeTriangleIndex = { TMP_CUBE_INDICES };
#undef TMP_CUBE_INDICES
                m_Private->m_CubeIndexRHI = m_Rhi->CreateBuffer("Ayanami.Persistent.ScreenProbe.CubeIndex",
                    sizeof(u32) * SizeCast<u32>(cubeTriangleIndex.size()),
                    RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_Index, false, false);

                auto stagingBuffer = m_Rhi->CreateStagedSingleBuffer(m_Private->m_CubeIndexRHI.get());
                auto tq            = m_Rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
                tq->RunSyncCommand([&](const RhiCommandList* cmdList) {
                    stagingBuffer->CmdCopyToDevice(
                        cmdList, cubeTriangleIndex.data(), sizeof(u32) * SizeCast<u32>(cubeTriangleIndex.size()), 0);
                });
            }
            m_Private->m_CubeIndex =
                &builder.ImportBuffer("Ayanami.RDG.ScreeProbe.CubeIndex", m_Private->m_CubeIndexRHI.get());
        }
        // meshdf culling texture
        {
            m_Private->m_MeshDFCullingDummy = &builder.DeclareTexture("Ayanami.RDG.ScreeProbe.MeshDFCullingDummy",
                FrameGraphTextureDesc(m_Private->m_MDFCullGridSizeXY, m_Private->m_MDFCullGridSizeXY, 1,
                    RhiImgFmt_R8_UNORM, RhiImageUsage::RhiImgUsage_RenderTarget));

            m_Private->m_MeshDFCullingDepth = &builder.DeclareTexture("Ayanami.RDG.ScreeProbe.MeshDFCullingDepth",
                FrameGraphTextureDesc(m_Private->m_MDFCullGridSizeXY, m_Private->m_MDFCullGridSizeXY, 1,
                    RhiImgFmt_D32_SFLOAT, RhiImageUsage::RhiImgUsage_Depth));

            m_Private->m_MeshDFCullingMatrix = &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.MeshDFCullingMatrix",
                FrameGraphBufferDesc(
                    sizeof(Matrix4x4f) * m_Private->m_MDFCullGridSizeZ, RhiBufferUsage::RhiBufferUsage_SSBO));

            u32 totalCullGrids =
                m_Private->m_MDFCullGridSizeXY * m_Private->m_MDFCullGridSizeXY * m_Private->m_MDFCullGridSizeZ;

            // warn: when enlarge, check if u32 overflows
            u32 totalCullGridsMDFSeats = totalCullGrids * m_Private->m_MDFMaxCullObjInGrid;

            m_Private->m_MeshDFCullingListCounter =
                &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.MeshDFCullingListCounter",
                    FrameGraphBufferDesc(
                        sizeof(u32) * totalCullGrids, RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage_CopyDst));
            m_Private->m_MeshDFCullingList = &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.MeshDFCullingList",
                FrameGraphBufferDesc(sizeof(u32) * totalCullGridsMDFSeats,
                    RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage_CopyDst));

            // warning: drawindirectindexed requires 5 args
            m_Private->m_MeshDFCullingIndirectArgs =
                &builder.DeclareBuffer("Ayanami.RDG.ScreeProbe.MeshDFCullingIndirectArgs",
                    FrameGraphBufferDesc(sizeof(u32) * 5,
                        RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_Indirect));
        }
    }
    IFRIT_APIDECL void AyanamiScreenProbeProcessor::AdaptiveScreenProbePlace(
        FrameGraphBuilder& builder, u32 perframeCBV, FGTextureNodeRef viewNormal, FGTextureNodeRef viewDepth)
    {
        AddClearUAVPass(
            builder, "Ayanami.ScreenProbe.ClearAdaptiveProbesCounter", *m_Private->m_AdaptiveProbesCounter, 0);

        struct PushConst
        {
            Vector2f m_CoordJitter;
            u32      m_PerFrameCBV;
            u32      m_ScrNormalCombSRV;
            u32      m_ScrDepthCombSRV;
            u32      m_AdaptiveProbesCounterUAV;
            u32      m_AdaptiveProbesListUAV;
            u32      m_DownSampleSize;
            u32      m_RTWidth;
            u32      m_RTHeight;
            u32      m_MaxAdaptiveProbes;
        } pc;

        pc.m_CoordJitter              = Vector2f(0.0f, 0.0f);
        pc.m_PerFrameCBV              = perframeCBV;
        pc.m_ScrNormalCombSRV         = 0;
        pc.m_ScrDepthCombSRV          = 0;
        pc.m_AdaptiveProbesCounterUAV = 0;
        pc.m_AdaptiveProbesListUAV    = 0;
        pc.m_DownSampleSize           = 4;
        pc.m_RTWidth                  = viewNormal->GetWidth();
        pc.m_RTHeight                 = viewNormal->GetHeight();
        pc.m_MaxAdaptiveProbes        = m_Private->m_MaxAdaptiveProbesCount;

        // update active rt size
        m_Private->m_ActiveRTWidth  = pc.m_RTWidth;
        m_Private->m_ActiveRTHeight = pc.m_RTHeight;

        // num of probes proposals
        i32 PropX = DivRoundUp(pc.m_RTWidth, pc.m_DownSampleSize);
        i32 PropY = DivRoundUp(pc.m_RTHeight, pc.m_DownSampleSize);

        i32 tgX = DivRoundUp(PropX, kAyanamiScrProbeAdaptivePlaceKernelSize);
        i32 tgY = DivRoundUp(PropY, kAyanamiScrProbeAdaptivePlaceKernelSize);

        AddComputePass<PushConst>(builder, "Ayanami.ScreenProbe.AdaptivePlace",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeAdaptivePlaceCS, {}),
            Vector3i{ (i32)tgX, (i32)tgY, 1 }, pc,
            [viewNormal, viewDepth, this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_ScrNormalCombSRV         = ctx.m_FgDesc->GetSRV(*viewNormal);
                data.m_ScrDepthCombSRV          = ctx.m_FgDesc->GetSRV(*viewDepth);
                data.m_AdaptiveProbesCounterUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesCounter);
                data.m_AdaptiveProbesListUAV    = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesList);
                SetRootConstant(data, ctx);
            })
            .AddReadResource(*viewNormal)
            .AddReadResource(*viewDepth)
            .AddReadWriteResource(*m_Private->m_AdaptiveProbesCounter)
            .AddWriteResource(*m_Private->m_AdaptiveProbesList);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::ProbeScreenTrace(
        FrameGraphBuilder& builder, u32 perframeCBV, FGBufferNodeRef hizBuffer, FGTextureNodeRef lastFrameFinalLighting)
    {
        AddClearUAVPass(builder, "Ayanami.ScreenProbe.ClearMeshDFTracingRayIndirectArgs",
            *m_Private->m_MeshDFTracingRayIndirectArgs, 0);

        struct PushConst
        {
            Vector2f m_RayJitter;
            u32      m_HizStorage;
            u32      m_PerFrameCBV;
            u32      m_RTWidth;
            u32      m_RTHeight;
            u32      m_ScreenProbeLightingAtlasUAV;
            u32      m_MeshDFTraceProposalCounterUAV;
            u32      m_MeshDFTraceProposalListUAV;
            u32      m_AdaptiveProbesListUAV;
            u32      m_LastFrameFinalLighting;
        } pc;

        pc.m_RayJitter                   = m_Private->m_RayJitter;
        pc.m_HizStorage                  = 0;
        pc.m_PerFrameCBV                 = perframeCBV;
        pc.m_RTWidth                     = m_Private->m_ActiveRTWidth;
        pc.m_RTHeight                    = m_Private->m_ActiveRTHeight;
        pc.m_ScreenProbeLightingAtlasUAV = 0;
        pc.m_MeshDFTraceProposalListUAV  = 0;
        pc.m_LastFrameFinalLighting      = 0;

        m_Private->m_ActiveGBufferAlbedo = lastFrameFinalLighting;

        // the indirect compute arg starts at offset 4
        AddIndirectComputePass<PushConst>(builder, "Ayanami.ScreenProbe.ScreenSpaceTrace",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeTraceScreenCS, {}),
            *m_Private->m_AdaptiveProbesCounter, 4 * sizeof(u32), pc,
            [hizBuffer, lastFrameFinalLighting, this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_HizStorage                    = ctx.m_FgDesc->GetSRV(*hizBuffer);
                data.m_ScreenProbeLightingAtlasUAV   = ctx.m_FgDesc->GetUAV(*m_Private->m_RadianceAtlas);
                data.m_MeshDFTraceProposalListUAV    = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFTracingRayList);
                data.m_MeshDFTraceProposalCounterUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFTracingRayIndirectArgs);
                data.m_AdaptiveProbesListUAV         = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesList);
                data.m_LastFrameFinalLighting        = ctx.m_FgDesc->GetSRV(*lastFrameFinalLighting);
                SetRootConstant(data, ctx);
            })
            .AddReadResource(*hizBuffer)
            .AddWriteResource(*m_Private->m_RadianceAtlas)
            .AddWriteResource(*m_Private->m_MeshDFTracingRayList)
            .AddReadResource(*m_Private->m_AdaptiveProbesCounter)
            .AddReadResource(*m_Private->m_AdaptiveProbesList)
            .AddReadResource(*lastFrameFinalLighting)
            .AddReadWriteResource(*m_Private->m_MeshDFTracingRayIndirectArgs);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::PrepareMeshDFCulling(
        FrameGraphBuilder& builder, u32 numTotalMdfs, Vector3f worldBoundMin, Vector3f worldBoundMax)
    {
        struct PushConst
        {
            Vector4f m_WorldBoundMin;
            Vector4f m_WorldBoundMax;
            u32      m_IndirectDrawArgs;
            u32      m_SlicesZ;
            u32      m_GridVpUAV;
            u32      m_TotalMdfCount;
        } pc;
        pc.m_WorldBoundMin    = Vector4f(worldBoundMin, 0.0f);
        pc.m_WorldBoundMax    = Vector4f(worldBoundMax, 0.0f);
        pc.m_IndirectDrawArgs = 0;
        pc.m_SlicesZ          = m_Private->m_MDFCullGridSizeZ;
        pc.m_GridVpUAV        = 0;
        pc.m_TotalMdfCount    = numTotalMdfs;

        m_Private->m_ActiveWorldBoundMax = pc.m_WorldBoundMax;
        m_Private->m_ActiveWorldBoundMin = pc.m_WorldBoundMin;
        m_Private->m_ActiveMDFCounts     = numTotalMdfs;

        auto tgX = DivRoundUp(m_Private->m_MDFCullGridSizeZ, kAyanamiScrProbeMDFCullPrepKernelSize);
        AddComputePass<PushConst>(builder, "Ayanami.ScreenProbe.MDFCullingPrep",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeMDFCullPrepCS, {}),
            Vector3i{ (i32)tgX, 1, 1 }, pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_GridVpUAV        = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFCullingMatrix);
                data.m_IndirectDrawArgs = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFCullingIndirectArgs);
                SetRootConstant(data, ctx);
            })
            .AddWriteResource(*m_Private->m_MeshDFCullingMatrix)
            .AddWriteResource(*m_Private->m_MeshDFCullingIndirectArgs);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::ScatterMeshDFToGrids(
        FrameGraphBuilder& builder, u32 perframeCBV, u32 numTotalMdfs, u32 meshDFDescUAV)
    {
        AddClearUAVPass(
            builder, "Ayanami.ScreenProbe.ClearMeshDFCullingListCounter", *m_Private->m_MeshDFCullingListCounter, 0);
        struct PushConst
        {
            Vector4f m_WorldBoundMin;
            Vector4f m_WorldBoundMax;
            u32      m_MeshDFDescListId;
            u32      m_PerFrameId;
            u32      m_TotalMdfCount;
            u32      m_GridVpUAV;
            u32      m_MaxMdfsPerGrid;
            u32      m_NumGridsPerSlice;
            u32      m_ScatterCounterUAV;
            u32      m_ScatterOutputUAV;
            u32      m_NumTilesWidth;
        } pc;
        pc.m_WorldBoundMin     = m_Private->m_ActiveWorldBoundMin;
        pc.m_WorldBoundMax     = m_Private->m_ActiveWorldBoundMax;
        pc.m_MeshDFDescListId  = meshDFDescUAV;
        pc.m_PerFrameId        = perframeCBV;
        pc.m_TotalMdfCount     = numTotalMdfs;
        pc.m_GridVpUAV         = 0;
        pc.m_MaxMdfsPerGrid    = m_Private->m_MDFMaxCullObjInGrid;
        pc.m_NumGridsPerSlice  = m_Private->m_MDFCullGridSizeXY * m_Private->m_MDFCullGridSizeXY;
        pc.m_ScatterCounterUAV = 0;
        pc.m_ScatterOutputUAV  = 0;
        pc.m_NumTilesWidth     = m_Private->m_MDFCullGridSizeXY;

        auto drawArgs       = GraphicsPassArgs{};
        drawArgs.m_CullMode = RhiCullMode::Front;

        AddIndirectDrawPass<PushConst>(builder, "Ayanami.ScreenProbe.MDFCullScatter",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeMDFCullScatterVS, {}),
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeMDFCullScatterFS, {}),
            *m_Private->m_MeshDFCullingIndirectArgs, *m_Private->m_CubeIndex, 0, drawArgs, pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_GridVpUAV         = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFCullingMatrix);
                data.m_ScatterCounterUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFCullingListCounter);
                data.m_ScatterOutputUAV  = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFCullingList);
                SetRootConstant(data, ctx);
            })
            .AddRenderTarget(*m_Private->m_MeshDFCullingDummy)
            //.AddDepthTarget(*m_Private->m_MeshDFCullingDepth, RhiRenderTargetLoadOp::ClearNoStore)
            .AddReadResource(*m_Private->m_MeshDFCullingIndirectArgs)
            .AddReadResource(*m_Private->m_CubeIndex)
            .AddWriteResource(*m_Private->m_MeshDFCullingMatrix)
            .AddWriteResource(*m_Private->m_MeshDFCullingListCounter)
            .AddWriteResource(*m_Private->m_MeshDFCullingList);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::ProbeMDFTrace(FrameGraphBuilder& builder, u32 perframeCBV,
        u32 meshDFDescUAV, FGTextureNodeRef gbufferDepth, u32 allCardDataId, FGTextureNodeRef cardDepthAtlas,
        FGTextureNodeRef cardAlbedoAtlas, u32 cardResolution, u32 cardAtlasResolution)
    {
        AddClearUAVPass(builder, "Ayanami.ScreenProbe.ClearGlobalDFTraceProposalCounter",
            *m_Private->m_GlobalDFTracingIndirectArgs, 0);

        struct PushConst
        {
            Vector4f m_WorldBoundMin;
            Vector4f m_WorldBoundMax;
            Vector4f m_CullGridSize;
            Vector2f m_RayJitter;
            u32      m_PerFrameCBV;
            u32      m_RTWidth;
            u32      m_RTHeight;
            u32      m_ScreenProbeLightingAtlasUAV;
            u32      m_MeshDFTraceProposalCounterUAV;
            u32      m_MeshDFTraceProposalListUAV;
            u32      m_AdaptiveProbesListUAV;
            u32      m_GBufferDepthSRV;
            u32      m_CullGridCounterUAV;
            u32      m_CullGridListUAV;
            u32      m_MeshDFDescListId;
            u32      m_MaxMdfsPerGrid;
            u32      m_GlobalDFTraceProposalCounterUAV;
            u32      m_GlobalDFTraceProposalListUAV;
            u32      m_GBufferAlbedoSRV;
            u32      m_NumMeshDFs;

            u32      m_AllCardObjDataId;
            u32      m_CardDepthAtlasSRV;
            u32      m_CardAlbedoAtlasSRV;
            u32      m_CardResolution;
            u32      m_CardAtlasResolution;
        } pc;
        pc.m_WorldBoundMin = m_Private->m_ActiveWorldBoundMin;
        pc.m_WorldBoundMax = m_Private->m_ActiveWorldBoundMax;
        pc.m_CullGridSize  = Vector4f(1.0f * m_Private->m_MDFCullGridSizeXY, 1.0f * m_Private->m_MDFCullGridSizeXY,
             1.0f * m_Private->m_MDFCullGridSizeZ, 0.0f);
        pc.m_RayJitter     = m_Private->m_RayJitter;
        pc.m_PerFrameCBV   = perframeCBV;
        pc.m_RTWidth       = m_Private->m_ActiveRTWidth;
        pc.m_RTHeight      = m_Private->m_ActiveRTHeight;
        pc.m_ScreenProbeLightingAtlasUAV     = 0;
        pc.m_MeshDFTraceProposalListUAV      = 0;
        pc.m_MeshDFTraceProposalCounterUAV   = 0;
        pc.m_AdaptiveProbesListUAV           = 0;
        pc.m_GBufferDepthSRV                 = 0;
        pc.m_CullGridCounterUAV              = 0;
        pc.m_CullGridListUAV                 = 0;
        pc.m_MeshDFDescListId                = meshDFDescUAV;
        pc.m_MaxMdfsPerGrid                  = m_Private->m_MDFMaxCullObjInGrid;
        pc.m_GlobalDFTraceProposalCounterUAV = 0;
        pc.m_GlobalDFTraceProposalListUAV    = 0;
        pc.m_NumMeshDFs                      = m_Private->m_ActiveMDFCounts;
        pc.m_GBufferAlbedoSRV                = 0;

        pc.m_AllCardObjDataId    = allCardDataId;
        pc.m_CardDepthAtlasSRV   = 0;
        pc.m_CardAlbedoAtlasSRV  = 0;
        pc.m_CardResolution      = cardResolution;
        pc.m_CardAtlasResolution = cardAtlasResolution;

        AddIndirectComputePass<PushConst>(builder, "Ayanami.ScreenProbe.MDFTrace",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeMDFTraceCS, {}),
            *m_Private->m_MeshDFTracingRayIndirectArgs, 1 * sizeof(u32), pc,
            [this, gbufferDepth, cardDepthAtlas, cardAlbedoAtlas](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_ScreenProbeLightingAtlasUAV   = ctx.m_FgDesc->GetUAV(*m_Private->m_RadianceAtlas);
                data.m_MeshDFTraceProposalListUAV    = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFTracingRayList);
                data.m_MeshDFTraceProposalCounterUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFTracingRayIndirectArgs);
                data.m_AdaptiveProbesListUAV         = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesList);
                data.m_GBufferDepthSRV               = ctx.m_FgDesc->GetSRV(*gbufferDepth);
                data.m_CullGridCounterUAV            = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFCullingListCounter);
                data.m_CullGridListUAV               = ctx.m_FgDesc->GetUAV(*m_Private->m_MeshDFCullingList);
                data.m_GBufferAlbedoSRV              = ctx.m_FgDesc->GetSRV(*m_Private->m_ActiveGBufferAlbedo);
                data.m_GlobalDFTraceProposalCounterUAV =
                    ctx.m_FgDesc->GetUAV(*m_Private->m_GlobalDFTracingIndirectArgs);
                data.m_GlobalDFTraceProposalListUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_GlobalDFTracingList);

                data.m_CardDepthAtlasSRV  = ctx.m_FgDesc->GetSRV(*cardDepthAtlas);
                data.m_CardAlbedoAtlasSRV = ctx.m_FgDesc->GetSRV(*cardAlbedoAtlas);
                SetRootConstant(data, ctx);
            })
            .AddReadResource(*m_Private->m_MeshDFTracingRayIndirectArgs)
            .AddReadResource(*m_Private->m_MeshDFTracingRayList)
            .AddReadResource(*m_Private->m_AdaptiveProbesList)
            .AddReadResource(*m_Private->m_MeshDFCullingListCounter)
            .AddReadResource(*m_Private->m_MeshDFCullingList)
            .AddReadResource(*gbufferDepth)
            .AddReadResource(*m_Private->m_ActiveGBufferAlbedo)
            .AddReadResource(*cardDepthAtlas)
            .AddReadResource(*cardAlbedoAtlas)
            .AddReadWriteResource(*m_Private->m_GlobalDFTracingIndirectArgs)
            .AddReadWriteResource(*m_Private->m_GlobalDFTracingList)
            .AddReadWriteResource(*m_Private->m_RadianceAtlas);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::ProbeGDFTrace(FrameGraphBuilder& builder, u32 perframeCBV,
        FGTextureNodeRef gbufferDepth, FGTextureNodeRef globalDF, u32 globalDFWSRange, u32 allCardDataId,
        FGTextureNodeRef cardDepthAtlas, FGTextureNodeRef cardAlbedoAtlas, u32 cardResolution, u32 cardAtlasResolution,
        u32 globalDFResolution, u32 voxelsPerClipMapwidth, FGBufferNodeRef globalObjectGrids, u32 meshDFDesc)
    {
        struct PushConst
        {
            Vector4f m_WorldBoundMin;
            Vector4f m_WorldBoundMax;
            Vector2f m_RayJitter;
            u32      m_GlobalDFSRV;
            u32      m_GlobalDFTraceProposalCounterUAV;
            u32      m_GlobalDFTraceProposalListUAV;
            u32      m_AdaptiveProbesListUAV;
            u32      m_PerFrameCBV;
            u32      m_RTWidth;
            u32      m_RTHeight;
            u32      m_GBufferDepthSRV;
            u32      m_GBufferAlbedoSRV;
            u32      m_ScreenProbeLightingAtlasUAV;

            u32      m_AllCardObjDataId;
            u32      m_CardDepthAtlasSRV;
            u32      m_CardAlbedoAtlasSRV;
            u32      m_CardResolution;
            u32      m_CardAtlasResolution;

            u32      m_GlobalDFResolution;
            u32      m_VoxelsPerClipMapwidth;
            u32      m_ObjectGridUAV;
            u32      m_MeshDFDescListId;
        } pc;
        pc.m_RayJitter                       = m_Private->m_RayJitter;
        pc.m_GlobalDFSRV                     = 0;
        pc.m_WorldBoundMin                   = Vector4f(-(f32)globalDFWSRange);
        pc.m_WorldBoundMax                   = Vector4f((f32)globalDFWSRange);
        pc.m_PerFrameCBV                     = perframeCBV;
        pc.m_RTWidth                         = m_Private->m_ActiveRTWidth;
        pc.m_RTHeight                        = m_Private->m_ActiveRTHeight;
        pc.m_GlobalDFTraceProposalCounterUAV = 0;
        pc.m_GlobalDFTraceProposalListUAV    = 0;
        pc.m_AdaptiveProbesListUAV           = 0;
        pc.m_GBufferDepthSRV                 = 0;
        pc.m_GlobalDFSRV                     = 0;
        pc.m_ScreenProbeLightingAtlasUAV     = 0;

        pc.m_AllCardObjDataId    = allCardDataId;
        pc.m_CardDepthAtlasSRV   = 0;
        pc.m_CardAlbedoAtlasSRV  = 0;
        pc.m_CardResolution      = cardResolution;
        pc.m_CardAtlasResolution = cardAtlasResolution;

        pc.m_GlobalDFResolution    = globalDFResolution;
        pc.m_VoxelsPerClipMapwidth = voxelsPerClipMapwidth;
        pc.m_ObjectGridUAV         = 0;
        pc.m_MeshDFDescListId      = meshDFDesc;

        AddIndirectComputePass<PushConst>(builder, "Ayanami.ScreenProbe.GDFTrace",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeGDFTraceCS, {}),
            *m_Private->m_GlobalDFTracingIndirectArgs, 1 * sizeof(u32), pc,
            [this, gbufferDepth, globalDF, cardDepthAtlas, cardAlbedoAtlas, globalObjectGrids](
                PushConst data, const FrameGraphPassContext& ctx) {
                data.m_GlobalDFSRV = ctx.m_FgDesc->GetSRV(*globalDF);
                data.m_GlobalDFTraceProposalCounterUAV =
                    ctx.m_FgDesc->GetUAV(*m_Private->m_GlobalDFTracingIndirectArgs);
                data.m_GlobalDFTraceProposalListUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_GlobalDFTracingList);
                data.m_AdaptiveProbesListUAV        = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesList);
                data.m_GBufferDepthSRV              = ctx.m_FgDesc->GetSRV(*gbufferDepth);
                data.m_GBufferAlbedoSRV             = ctx.m_FgDesc->GetSRV(*m_Private->m_ActiveGBufferAlbedo);
                data.m_ScreenProbeLightingAtlasUAV  = ctx.m_FgDesc->GetUAV(*m_Private->m_RadianceAtlas);

                data.m_CardDepthAtlasSRV  = ctx.m_FgDesc->GetSRV(*cardDepthAtlas);
                data.m_CardAlbedoAtlasSRV = ctx.m_FgDesc->GetSRV(*cardAlbedoAtlas);

                data.m_ObjectGridUAV = ctx.m_FgDesc->GetUAV(*globalObjectGrids);
                SetRootConstant(data, ctx);
            })
            .AddReadResource(*gbufferDepth)
            .AddReadResource(*globalDF)
            .AddReadResource(*m_Private->m_AdaptiveProbesList)
            .AddWriteResource(*m_Private->m_RadianceAtlas)
            .AddReadResource(*m_Private->m_ActiveGBufferAlbedo)
            .AddReadResource(*cardAlbedoAtlas)
            .AddReadResource(*cardDepthAtlas)
            .AddReadResource(*globalObjectGrids)
            .AddReadWriteResource(*m_Private->m_GlobalDFTracingIndirectArgs)
            .AddReadWriteResource(*m_Private->m_GlobalDFTracingList);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::ProbeOctMappingBorderFix(FrameGraphBuilder& builder)
    {

        struct PushConst
        {
            u32 m_AdaptiveProbesCounterUAV;
            u32 m_RTWidth;
            u32 m_RTHeight;
            u32 m_ScreenProbeLightingAtlasUAVIn;
            u32 m_ScreenProbeLightingAtlasUAVOut;
        } pc;
        pc.m_AdaptiveProbesCounterUAV       = 0;
        pc.m_RTWidth                        = m_Private->m_ActiveRTWidth;
        pc.m_RTHeight                       = m_Private->m_ActiveRTHeight;
        pc.m_ScreenProbeLightingAtlasUAVIn  = 0;
        pc.m_ScreenProbeLightingAtlasUAVOut = 0;

        AddIndirectComputePass<PushConst>(builder, "Ayanami.ScreenProbe.OctMappingBorderFix",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeBorderFixCS, {}),
            *m_Private->m_AdaptiveProbesCounter, 7 * sizeof(u32), pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_AdaptiveProbesCounterUAV       = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesCounter);
                data.m_ScreenProbeLightingAtlasUAVIn  = ctx.m_FgDesc->GetUAV(*m_Private->m_RadianceAtlas);
                data.m_ScreenProbeLightingAtlasUAVOut = ctx.m_FgDesc->GetUAV(*m_Private->m_RadianceAtlasFixed);
                SetRootConstant(data, ctx);
            })
            .AddReadResource(*m_Private->m_AdaptiveProbesCounter)
            .AddReadResource(*m_Private->m_RadianceAtlas)
            .AddWriteResource(*m_Private->m_RadianceAtlasFixed);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::ProbeIntegrate(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            Vector2f m_RayJitter;
            u32      m_AdaptiveProbesCounterUAV;
            u32      m_AdaptiveProbesListUAV;
            u32      m_RTWidth;
            u32      m_RTHeight;
            u32      m_ScreenProbeLightingAtlasUAV;
            u32      m_OutputSHCoefBufferUAV;
        } pc;
        pc.m_RayJitter                   = m_Private->m_RayJitter;
        pc.m_AdaptiveProbesCounterUAV    = 0;
        pc.m_AdaptiveProbesListUAV       = 0;
        pc.m_RTWidth                     = m_Private->m_ActiveRTWidth;
        pc.m_RTHeight                    = m_Private->m_ActiveRTHeight;
        pc.m_ScreenProbeLightingAtlasUAV = 0;
        pc.m_OutputSHCoefBufferUAV       = 0;

        AddIndirectComputePass<PushConst>(builder, "Ayanami.ScreenProbe.IntegrateSH",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbeSHIntegrateCS, {}),
            *m_Private->m_AdaptiveProbesCounter, 7 * sizeof(u32), pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_AdaptiveProbesCounterUAV    = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesCounter);
                data.m_AdaptiveProbesListUAV       = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesList);
                data.m_ScreenProbeLightingAtlasUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_RadianceAtlasFixed);
                data.m_OutputSHCoefBufferUAV       = ctx.m_FgDesc->GetUAV(*m_Private->m_IntegratedSH);
                SetRootConstant(data, ctx);
            })
            .AddReadResource(*m_Private->m_AdaptiveProbesCounter)
            .AddReadResource(*m_Private->m_AdaptiveProbesList)
            .AddReadResource(*m_Private->m_RadianceAtlasFixed)
            .AddReadWriteResource(*m_Private->m_IntegratedSH);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::ProbePixelGather(FrameGraphBuilder& builder, u32 perframeCBV,
        FGTextureNodeRef gbufferDepth, FGTextureNodeRef gbufferNormal, FGTextureNodeRef outputTex)
    {
        struct PushConst
        {
            u32 m_RTWidth;
            u32 m_RTHeight;
            u32 m_PerFrameCBV;
            u32 m_ScrNormalCombSRV;
            u32 m_ScrDepthCombSRV;
            u32 m_OutputSHCoefBufferUAV;
            u32 m_OutTexUAV;
        } pc;
        pc.m_RTWidth               = m_Private->m_ActiveRTWidth;
        pc.m_RTHeight              = m_Private->m_ActiveRTHeight;
        pc.m_PerFrameCBV           = perframeCBV;
        pc.m_ScrNormalCombSRV      = 0;
        pc.m_ScrDepthCombSRV       = 0;
        pc.m_OutputSHCoefBufferUAV = 0;
        pc.m_OutTexUAV             = 0;

        auto tgX = DivRoundUp(pc.m_RTWidth, kAyanamiScrProbePixelGatherKernelSize);
        auto tgY = DivRoundUp(pc.m_RTHeight, kAyanamiScrProbePixelGatherKernelSize);

        AddComputePass<PushConst>(builder, "Ayanami.ScreenProbe.PixelGather",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.ScreenProbePixelGatherCS, {}),
            Vector3i{ (i32)tgX, (i32)tgY, 1 }, pc,
            [this, gbufferDepth, gbufferNormal, outputTex](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_ScrNormalCombSRV      = ctx.m_FgDesc->GetSRV(*gbufferNormal);
                data.m_ScrDepthCombSRV       = ctx.m_FgDesc->GetSRV(*gbufferDepth);
                data.m_OutputSHCoefBufferUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_IntegratedSH);
                data.m_OutTexUAV             = ctx.m_FgDesc->GetUAV(*outputTex);
                SetRootConstant(data, ctx);
            })
            .AddReadResource(*gbufferDepth)
            .AddReadResource(*gbufferNormal)
            .AddWriteResource(*outputTex)
            .AddReadResource(*m_Private->m_IntegratedSH);
    }

    IFRIT_APIDECL void AyanamiScreenProbeProcessor::FrameProceed()
    {
        // todo
        auto curFrameId        = m_SharedContext->m_FrameIdx % 32;
        auto jitter            = Hammersley2d(SizeCast<u32>(curFrameId), 32);
        jitter                 = jitter - Vector2f(0.5f, 0.5f); // center jitter
        m_Private->m_RayJitter = jitter;
    }

    IFRIT_APIDECL FGBufferNodeRef AyanamiScreenProbeProcessor::GetAdaptiveProbesList() const
    {
        return m_Private->m_AdaptiveProbesList;
    }
    IFRIT_APIDECL FGBufferNodeRef AyanamiScreenProbeProcessor::GetAdaptiveProbesCounter() const
    {
        return m_Private->m_AdaptiveProbesCounter;
    }
    IFRIT_APIDECL FGTextureNodeRef AyanamiScreenProbeProcessor::GetScreenProbeRadianceAtlas() const
    {
        return m_Private->m_RadianceAtlas;
    }
} // namespace Ifrit::Runtime::Ayanami