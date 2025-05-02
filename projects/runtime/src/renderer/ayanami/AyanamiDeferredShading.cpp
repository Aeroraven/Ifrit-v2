
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

#include "ifrit/runtime/renderer/ayanami/AyanamiDeferredShading.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Ayanami.h"

using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime::FrameGraphUtils;

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiDeferredShadingPrivate
    {
        IF_CONSTEXPR static u32 kHistoryFrames = 8;

        u32                     m_ActiveRTWidth  = 0;
        u32                     m_ActiveRTHeight = 0;
        u32                     m_FrameIdx       = 0;

        FGTextureNodeRef        m_DeferredShadowTexture         = nullptr;
        FGTextureNodeRef        m_DeferredDirectLightingTexture = nullptr;

        FGTextureNodeRef        m_LastFrameFinalLightingTex   = nullptr;
        FGTextureNodeRef        m_CurFrameFinalLightingTex    = nullptr;
        FGTextureNodeRef        m_CurFrameIndirectLightingTex = nullptr;

        // Managed Resources (Persistent)
        bool                    m_PersistentResourceInited = false;
        RhiTextureRef           m_IndirectLightingTex[2];
        RhiTextureRef           m_FinalLightingTex[kHistoryFrames];
    };
    AyanamiDeferredShading::AyanamiDeferredShading(Graphics::Rhi::RhiBackend* rhi)
        : m_Private(new AyanamiDeferredShadingPrivate()), m_Rhi(rhi)
    {
    }
    AyanamiDeferredShading::~AyanamiDeferredShading() { delete m_Private; }

    void AyanamiDeferredShading::InitContext(FrameGraphBuilder& builder, u32 rtWidth, u32 rtHeight)
    {
        m_Private->m_ActiveRTWidth  = rtWidth;
        m_Private->m_ActiveRTHeight = rtHeight;
        m_Private->m_FrameIdx       = (m_Private->m_FrameIdx + 1) % m_Private->kHistoryFrames;

        if (!m_Private->m_PersistentResourceInited)
        {
            m_Private->m_IndirectLightingTex[0] = m_Rhi->CreateTexture2D("Ayanami.Tex.IndirectLighting0", rtWidth,
                rtHeight, RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead
                    | RhiImageUsage::RhiImgUsage_UnorderedAccess,
                true);
            m_Private->m_IndirectLightingTex[1] = m_Rhi->CreateTexture2D("Ayanami.Tex.IndirectLighting1", rtWidth,
                rtHeight, RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead
                    | RhiImageUsage::RhiImgUsage_UnorderedAccess,
                true);

            for (u32 i = 0; i < m_Private->kHistoryFrames; ++i)
            {
                m_Private->m_FinalLightingTex[i] =
                    m_Rhi->CreateTexture2D("Ayanami.Tex.FinalLighting" + std::to_string(i), rtWidth, rtHeight,
                        RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                        RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead
                            | RhiImageUsage::RhiImgUsage_UnorderedAccess,
                        true);
            }
            m_Private->m_PersistentResourceInited = true;
        }

        m_Private->m_DeferredShadowTexture = &builder.DeclareTexture("Ayanami.RDG.FinalLighting.DirectShadow",
            FrameGraphTextureDesc(rtWidth, rtHeight, 1, Graphics::Rhi::RhiImageFormat::RhiImgFmt_R8_UNORM,
                Graphics::Rhi::RhiImageUsage::RhiImgUsage_RenderTarget
                    | Graphics::Rhi::RhiImageUsage::RhiImgUsage_ShaderRead));

        m_Private->m_DeferredDirectLightingTexture = &builder.DeclareTexture("Ayanami.RDG.FinalLighting.DirectLighting",
            FrameGraphTextureDesc(rtWidth, rtHeight, 1, Graphics::Rhi::RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                Graphics::Rhi::RhiImageUsage::RhiImgUsage_RenderTarget
                    | Graphics::Rhi::RhiImageUsage::RhiImgUsage_ShaderRead));

        m_Private->m_CurFrameFinalLightingTex = &builder.ImportTexture(
            "Ayanami.RDG.FinalLighting.CurFrame", m_Private->m_FinalLightingTex[m_Private->m_FrameIdx].get());
        m_Private->m_LastFrameFinalLightingTex   = &builder.ImportTexture("Ayanami.RDG.FinalLighting.LastFrame",
              m_Private
                  ->m_FinalLightingTex[(m_Private->m_FrameIdx + m_Private->kHistoryFrames - 1)
                    % m_Private->kHistoryFrames]
                  .get());
        m_Private->m_CurFrameIndirectLightingTex = &builder.ImportTexture(
            "Ayanami.RDG.IndirectLighting.CurFrame", m_Private->m_IndirectLightingTex[m_Private->m_FrameIdx % 2].get());
    }

    IFRIT_APIDECL void AyanamiDeferredShading::RenderDeferredShadow(FrameGraphBuilder& builder, u32 perFrameCBV,
        FGBufferNodeRef shadowData, FGTextureNodeRef gbufferDepth, FGTextureNodeRef gbufferNormal, u32 totalLights)
    {
        struct PushConst
        {
            u32 m_PerFrameCBV;
            u32 m_TotalLights;
            u32 m_LightDataId;
            u32 m_GBufferDepthSRV;
            u32 m_GBufferNormalSRV;
        } pc;

        pc.m_PerFrameCBV     = perFrameCBV;
        pc.m_TotalLights     = totalLights;
        pc.m_LightDataId     = 0;
        pc.m_GBufferDepthSRV = 0;

        AddPostProcessPass<PushConst>(builder, "Ayanami.FinalLighting.DirectShadow",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.DeferredShadowFS, {}), pc,
            [this, gbufferDepth, shadowData, gbufferNormal](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_LightDataId      = ctx.m_FgDesc->GetUAV(*shadowData);
                data.m_GBufferDepthSRV  = ctx.m_FgDesc->GetSRV(*gbufferDepth);
                data.m_GBufferNormalSRV = ctx.m_FgDesc->GetSRV(*gbufferNormal);
                SetRootSignature(data, ctx);
            })
            .AddRenderTarget(*m_Private->m_DeferredShadowTexture)
            .AddReadResource(*gbufferDepth)
            .AddReadResource(*gbufferNormal)
            .AddReadResource(*shadowData);
    }

    IFRIT_APIDECL void AyanamiDeferredShading::RenderDeferredLighting(FrameGraphBuilder& builder, u32 perFrameCBV,
        FGTextureNodeRef gbufferDepth, FGTextureNodeRef gbufferNormal, FGTextureNodeRef gbufferAlbedo,
        FGBufferNodeRef shadowData, u32 totalLights)
    {
        struct PushConst
        {
            u32 m_PerFrameCBV;
            u32 m_ShadowOcclusionSRV;
            u32 m_GAlbedoSRV;
            u32 m_GNormalSRV;
            u32 m_GDepthSRV;
            u32 m_TotalLights;
            u32 m_LightDataId;
        } pc;
        pc.m_PerFrameCBV        = perFrameCBV;
        pc.m_ShadowOcclusionSRV = 0;
        pc.m_GAlbedoSRV         = 0;
        pc.m_GNormalSRV         = 0;
        pc.m_GDepthSRV          = 0;
        pc.m_TotalLights        = totalLights;
        pc.m_LightDataId        = 0;

        AddPostProcessPass<PushConst>(builder, "Ayanami.FinalLighting.DirectLighting",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.DeferredLightingFS, {}), pc,
            [this, gbufferDepth, gbufferNormal, gbufferAlbedo, shadowData](
                PushConst data, const FrameGraphPassContext& ctx) {
                data.m_ShadowOcclusionSRV = ctx.m_FgDesc->GetSRV(*m_Private->m_DeferredShadowTexture);
                data.m_GAlbedoSRV         = ctx.m_FgDesc->GetSRV(*gbufferAlbedo);
                data.m_GNormalSRV         = ctx.m_FgDesc->GetSRV(*gbufferNormal);
                data.m_GDepthSRV          = ctx.m_FgDesc->GetSRV(*gbufferDepth);
                data.m_LightDataId        = ctx.m_FgDesc->GetUAV(*shadowData);
                SetRootSignature(data, ctx);
            })
            .AddRenderTarget(*m_Private->m_DeferredDirectLightingTexture)
            .AddReadResource(*gbufferDepth)
            .AddReadResource(*gbufferNormal)
            .AddReadResource(*gbufferAlbedo)
            .AddReadResource(*m_Private->m_DeferredShadowTexture)
            .AddReadResource(*shadowData);
    }

    IFRIT_APIDECL void AyanamiDeferredShading::ExperimentalFuse(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_IndirectLightingSRV;
            u32 m_DirectLightingSRV;
        } pc;

        pc.m_IndirectLightingSRV = 0;
        pc.m_DirectLightingSRV   = 0;

        AddPostProcessPass<PushConst>(builder, "Ayanami.FinalLighting.Fuse",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.DeferredExpMixFS, {}), pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_IndirectLightingSRV = ctx.m_FgDesc->GetSRV(*m_Private->m_CurFrameIndirectLightingTex);
                data.m_DirectLightingSRV   = ctx.m_FgDesc->GetSRV(*m_Private->m_DeferredDirectLightingTexture);
                SetRootSignature(data, ctx);
            })
            .AddRenderTarget(*m_Private->m_CurFrameFinalLightingTex)
            .AddReadResource(*m_Private->m_CurFrameIndirectLightingTex)
            .AddReadResource(*m_Private->m_DeferredDirectLightingTexture);
    }
    IFRIT_APIDECL FGTextureNodeRef AyanamiDeferredShading::GetRDGDirectShadowTexture() const
    {
        return m_Private->m_DeferredShadowTexture;
    }
    IFRIT_APIDECL FGTextureNodeRef AyanamiDeferredShading::GetRDGDirectLightingTexture() const
    {
        return m_Private->m_DeferredDirectLightingTexture;
    }
    IFRIT_APIDECL FGTextureNodeRef AyanamiDeferredShading::GetRDGLastFrameFinalLightingTexture() const
    {
        return m_Private->m_LastFrameFinalLightingTex;
    }
    IFRIT_APIDECL FGTextureNodeRef AyanamiDeferredShading::GetRDGFinalLightingTexture() const
    {
        return m_Private->m_CurFrameFinalLightingTex;
    }
} // namespace Ifrit::Runtime::Ayanami
