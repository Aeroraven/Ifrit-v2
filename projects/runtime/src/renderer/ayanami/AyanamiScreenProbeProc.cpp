
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
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Ayanami.h"

using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime::FrameGraphUtils;
using namespace Ifrit::Runtime::Ayanami::Config;

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiScreenProbeProcessorPrivate
    {
        f32              m_AdaptiveProbesRatio      = 0.0f;
        u32              m_MaxRTWidth               = 0;
        u32              m_MaxRTHeight              = 0;
        u32              m_MaxUniformTilesPerWidth  = 0;
        u32              m_MaxUniformTilesPerHeight = 0;
        u32              m_MaxUniformProbes         = 0;
        u32              m_MaxAdaptiveProbesCount   = 0;

        FGTextureNodeRef m_RadianceAtlas         = nullptr;
        FGTextureNodeRef m_ProbeSH_R             = nullptr;
        FGTextureNodeRef m_ProbeSH_G             = nullptr;
        FGTextureNodeRef m_ProbeSH_B             = nullptr;
        FGBufferNodeRef  m_AdaptiveProbesList    = nullptr;
        FGBufferNodeRef  m_AdaptiveProbesCounter = nullptr;
    };

    IFRIT_APIDECL AyanamiScreenProbeProcessor::AyanamiScreenProbeProcessor(Graphics::Rhi::RhiBackend* rhi) : m_Rhi(rhi)
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

        // Then, the probe sh
        {
            u32 requiredHeight         = m_Private->m_MaxUniformTilesPerHeight * kAyanami_ScreenProbeProbeHemiRes;
            u32 requiredWidth          = m_Private->m_MaxUniformTilesPerWidth * kAyanami_ScreenProbeProbeHemiRes;
            u32 requiredHeightAdaptive = static_cast<u32>(std::ceil(m_Private->m_AdaptiveProbesRatio * requiredHeight));
            u32 requiredHeightFinal    = requiredHeight + requiredHeightAdaptive;
            u32 requiredWidthFinal     = requiredWidth;

            m_Private->m_ProbeSH_R = &builder.DeclareTexture("Ayanami.RDG.ScreeProbe.ProbeSH_R",
                FrameGraphTextureDesc(requiredWidthFinal, requiredHeightFinal, 1, RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_ShaderRead));
            m_Private->m_ProbeSH_G = &builder.DeclareTexture("Ayanami.RDG.ScreeProbe.ProbeSH_G",
                FrameGraphTextureDesc(requiredWidthFinal, requiredHeightFinal, 1, RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_ShaderRead));
            m_Private->m_ProbeSH_B = &builder.DeclareTexture("Ayanami.RDG.ScreeProbe.ProbeSH_B",
                FrameGraphTextureDesc(requiredWidthFinal, requiredHeightFinal, 1, RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_ShaderRead));
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

        // num of probes proposals
        i32 PropX = DivRoundUp(pc.m_RTWidth, pc.m_DownSampleSize);
        i32 PropY = DivRoundUp(pc.m_RTHeight, pc.m_DownSampleSize);

        i32 tgX = DivRoundUp(PropX, kAyanamiScrProbeAdaptivePlaceKernelSize);
        i32 tgY = DivRoundUp(PropY, kAyanamiScrProbeAdaptivePlaceKernelSize);

        AddComputePass<PushConst>(builder, "Ayanami.ScreenProbe.AdaptivePlace",
            Internal::kIntShaderTableAyanami.ScreenProbeAdaptivePlaceCS, Vector3i{ (i32)tgX, (i32)tgY, 1 }, pc,
            [viewNormal, viewDepth, this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_ScrNormalCombSRV         = ctx.m_FgDesc->GetSRV(*viewNormal);
                data.m_ScrDepthCombSRV          = ctx.m_FgDesc->GetSRV(*viewDepth);
                data.m_AdaptiveProbesCounterUAV = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesCounter);
                data.m_AdaptiveProbesListUAV    = ctx.m_FgDesc->GetUAV(*m_Private->m_AdaptiveProbesList);
                SetRootSignature(data, ctx);
            })
            .AddReadResource(*viewNormal)
            .AddReadResource(*viewDepth)
            .AddReadWriteResource(*m_Private->m_AdaptiveProbesCounter)
            .AddWriteResource(*m_Private->m_AdaptiveProbesList);
    }

    IFRIT_APIDECL FGBufferNodeRef AyanamiScreenProbeProcessor::GetAdaptiveProbesList() const
    {
        return m_Private->m_AdaptiveProbesList;
    }
    IFRIT_APIDECL FGBufferNodeRef AyanamiScreenProbeProcessor::GetAdaptiveProbesCounter() const
    {
        return m_Private->m_AdaptiveProbesCounter;
    }
} // namespace Ifrit::Runtime::Ayanami