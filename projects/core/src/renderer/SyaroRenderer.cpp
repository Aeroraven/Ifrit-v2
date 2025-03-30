
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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

#include "ifrit/common/base/IfritBase.h"

#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/renderer/RendererUtil.h"
#include "ifrit/core/renderer/SyaroRenderer.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"
#include <algorithm>
#include <bit>

#include "ifrit.shader/Syaro/Syaro.SharedConst.h"

using namespace Ifrit::Graphics::Rhi;
using Ifrit::Common::Utility::SizeCast;
using Ifrit::Math::DivRoundUp;

// Frequently used image usages
IF_CONSTEXPR auto kbImUsage_UAV_SRV = RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_CopyDest =
    RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_TRANSFER_DST_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_RT_CopySrc = RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT
    | RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | RHI_IMAGE_USAGE_TRANSFER_SRC_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_RT =
    RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
IF_CONSTEXPR auto kbImUsage_SRV_DEPTH = RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
IF_CONSTEXPR auto kbImUsage_UAV       = RHI_IMAGE_USAGE_STORAGE_BIT;

// Frequently used buffer usages
IF_CONSTEXPR auto kbBufUsage_Indirect      = RhiBufferUsage_Indirect | RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO;
IF_CONSTEXPR auto kbBufUsage_SSBO_CopyDest = RhiBufferUsage_SSBO | RhiBufferUsage_CopyDst;
IF_CONSTEXPR auto kbBufUsage_SSBO          = RhiBufferUsage_SSBO;

// Frequently used image fmts
IF_CONSTEXPR auto kbImFmt_RGBA32F = RhiImgFmt_R32G32B32A32_SFLOAT;
IF_CONSTEXPR auto kbImFmt_RGBA16F = RhiImgFmt_R16G16B16A16_SFLOAT;
IF_CONSTEXPR auto kbImFmt_RG32F   = RhiImgFmt_R32G32_SFLOAT;
IF_CONSTEXPR auto kbImFmt_R32F    = RhiImgFmt_R32_SFLOAT;

namespace Ifrit::Core
{

    struct GPUHiZDesc
    {
        u32 m_width;
        u32 m_height;
    };

    std::vector<RhiResourceBarrier> RegisterUAVBarriers(
        const std::vector<RhiBuffer*>& buffers, const std::vector<RhiTexture*>& textures)
    {
        std::vector<RhiResourceBarrier> barriers;
        for (auto& buffer : buffers)
        {
            RhiUAVBarrier barrier;
            barrier.m_type   = RhiResourceType::Buffer;
            barrier.m_buffer = buffer;
            RhiResourceBarrier resBarrier;
            resBarrier.m_type = RhiBarrierType::UAVAccess;
            resBarrier.m_uav  = barrier;
            barriers.push_back(resBarrier);
        }
        for (auto& texture : textures)
        {
            RhiUAVBarrier barrier;
            barrier.m_type    = RhiResourceType::Texture;
            barrier.m_texture = texture;
            RhiResourceBarrier resBarrier;
            resBarrier.m_type = RhiBarrierType::UAVAccess;
            resBarrier.m_uav  = barrier;
            barriers.push_back(resBarrier);
        }
        return barriers;
    }

    void runImageBarrier(
        const RhiCommandList* cmd, RhiTexture* texture, RhiResourceState dst, RhiImageSubResource subResource)
    {
        std::vector<RhiResourceBarrier> barriers;
        RhiTransitionBarrier            barrier;
        barrier.m_type        = RhiResourceType::Texture;
        barrier.m_texture     = texture;
        barrier.m_srcState    = RhiResourceState::AutoTraced;
        barrier.m_dstState    = dst;
        barrier.m_subResource = subResource;

        RhiResourceBarrier resBarrier;
        resBarrier.m_type       = RhiBarrierType::Transition;
        resBarrier.m_transition = barrier;

        barriers.push_back(resBarrier);
        cmd->AddResourceBarrier(barriers);
    }

    void runUAVBufferBarrier(const RhiCommandList* cmd, RhiBuffer* buffer)
    {
        std::vector<RhiResourceBarrier> barriers;
        RhiUAVBarrier                   barrier;
        barrier.m_type   = RhiResourceType::Buffer;
        barrier.m_buffer = buffer;
        RhiResourceBarrier resBarrier;
        resBarrier.m_type = RhiBarrierType::UAVAccess;
        resBarrier.m_uav  = barrier;
        barriers.push_back(resBarrier);
        cmd->AddResourceBarrier(barriers);
    }

    RhiScissor getSupersampleDownsampledArea(const RhiRenderTargets* finalRenderTargets, const RendererConfig& cfg)
    {
        RhiScissor scissor;
        scissor.x      = 0;
        scissor.y      = 0;
        scissor.width  = finalRenderTargets->GetRenderArea().width / cfg.m_superSamplingRate;
        scissor.height = finalRenderTargets->GetRenderArea().height / cfg.m_superSamplingRate;
        return scissor;
    }

    // end of util functions

    IFRIT_APIDECL PerFrameData::PerViewData& SyaroRenderer::GetPrimaryView(PerFrameData& perframeData)
    {
        for (auto& view : perframeData.m_views)
        {
            if (view.m_viewType == PerFrameData::ViewType::Primary)
            {
                return view;
            }
        }
        throw std::runtime_error("Primary view not found");
        return perframeData.m_views[0];
    }
    IFRIT_APIDECL void SyaroRenderer::CreateTimer()
    {
        auto rhi        = m_app->GetRhi();
        auto timer      = rhi->CreateDeviceTimer();
        m_timer         = timer;
        auto deferTimer = rhi->CreateDeviceTimer();
        m_timerDefer    = deferTimer;
    }

    IFRIT_APIDECL SyaroRenderer::GPUShader* SyaroRenderer::CreateShaderFromFile(
        const std::string& shaderPath, const std::string& entry, Graphics::Rhi::RhiShaderStage stage)
    {
        auto              rhi            = m_app->GetRhi();
        std::string       shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
        auto              path           = shaderBasePath + "/Syaro/" + shaderPath;
        auto              shaderCode     = Ifrit::Common::Utility::ReadTextFile(path);
        std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
        return rhi->CreateShader(shaderPath, shaderCodeVec, entry, stage, RhiShaderSourceType::GLSLCode);
    }

    IFRIT_APIDECL void SyaroRenderer::SetupPostprocessPassAndTextures()
    {
        // passes
        m_acesToneMapping = std::make_unique<PostprocessPassCollection::PostFxAcesToneMapping>(m_app);
        m_globalFogPass   = std::make_unique<PostprocessPassCollection::PostFxGlobalFog>(m_app);
        m_gaussianHori    = std::make_unique<PostprocessPassCollection::PostFxGaussianHori>(m_app);
        m_gaussianVert    = std::make_unique<PostprocessPassCollection::PostFxGaussianVert>(m_app);
        m_fftConv2d       = std::make_unique<PostprocessPassCollection::PostFxFFTConv2d>(m_app);

        m_jointBilateralFilter = std::make_unique<PostprocessPassCollection::PostFxJointBilaterialFilter>(m_app);

        // tex and samplers
        m_postprocTexSampler = m_app->GetRhi()->CreateTrivialSampler();

        // fsr2
        m_fsr2proc = m_app->GetRhi()->CreateFsr2Processor();
    }

    IFRIT_APIDECL void SyaroRenderer::CreatePostprocessTextures(u32 width, u32 height)
    {
        auto rhi   = m_app->GetRhi();
        auto rtFmt = kbImFmt_RGBA32F;
        if (m_postprocTex.find({ width, height }) != m_postprocTex.end())
        {
            return;
        }
        for (u32 i = 0; i < 2; i++)
        {
            auto tex = rhi->CreateTexture2D("Syaro_PostprocTex", width, height, rtFmt, kbImUsage_UAV_SRV_RT, true);
            auto colorRT =
                rhi->CreateRenderTarget(tex.get(), { { 0.0f, 0.0f, 0.0f, 1.0f } }, RhiRenderTargetLoadOp::Load, 0, 0);
            auto rts = rhi->CreateRenderTargets();

            m_postprocTex[{ width, height }][i] = tex;
            m_postprocTexSRV[{ width, height }][i] =
                rhi->RegisterCombinedImageSampler(tex.get(), m_postprocTexSampler.get());
            m_postprocColorRT[{ width, height }][i] = colorRT;

            RhiAttachmentBlendInfo blendInfo;
            blendInfo.m_blendEnable         = true;
            blendInfo.m_srcColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
            blendInfo.m_dstColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blendInfo.m_colorBlendOp        = RhiBlendOp::RHI_BLEND_OP_ADD;
            blendInfo.m_alphaBlendOp        = RhiBlendOp::RHI_BLEND_OP_ADD;
            blendInfo.m_srcAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
            blendInfo.m_dstAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

            rts->SetColorAttachments({ colorRT.get() });
            rts->SetRenderArea({ 0, 0, width, height });
            colorRT->SetBlendInfo(blendInfo);
            m_postprocRTs[{ width, height }][i] = rts;
        }
    }

    IFRIT_APIDECL void SyaroRenderer::SetupAndRunFrameGraph(
        PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        // some pipelines
        if (m_deferredShadowPass == nullptr)
        {
            auto rhi         = m_app->GetRhi();
            auto vsShader    = CreateShaderFromFile("Syaro.DeferredShadow.vert.glsl", "main", RhiShaderStage::Vertex);
            auto fsShader    = CreateShaderFromFile("Syaro.DeferredShadow.frag.glsl", "main", RhiShaderStage::Fragment);
            auto shadowRtCfg = renderTargets->GetFormat();
            shadowRtCfg.m_colorFormats = { RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT };
            shadowRtCfg.m_depthFormat  = RhiImageFormat::RhiImgFmt_UNDEFINED;

            m_deferredShadowPass = rhi->CreateGraphicsPass();
            m_deferredShadowPass->SetVertexShader(vsShader);
            m_deferredShadowPass->SetPixelShader(fsShader);
            m_deferredShadowPass->SetNumBindlessDescriptorSets(3);
            m_deferredShadowPass->SetPushConstSize(3 * u32Size);
            m_deferredShadowPass->SetRenderTargetFormat(shadowRtCfg);
        }

        // binding resources to pass
        auto& primaryView  = GetPrimaryView(perframeData);
        auto  rhi          = m_app->GetRhi();
        auto  mainRtWidth  = primaryView.m_renderWidth;
        auto  mainRtHeight = primaryView.m_renderHeight;
        CreatePostprocessTextures(mainRtWidth, mainRtHeight);

        // declare frame graph
        FrameGraphBuilder  fg;

        Vec<ResourceNode*> resShadowMapTexs;
        Vec<u32>           shadowMapTexIds;

        for (auto id = 0; auto& view : perframeData.m_views)
        {
            if (view.m_viewType == PerFrameData::ViewType::Shadow)
            {
                auto& resId = fg.AddResource("ShadowMapTex" + std::to_string(id))
                                  .SetImportedResource(view.m_visibilityDepth_Combined.get(), { 0, 0, 1, 1 });
                resShadowMapTexs.push_back(&resId);
                shadowMapTexIds.push_back(id);
            }
            id++;
        }

        // Some Resources
        auto& resAtmosphereOutput =
            fg.AddResource("AtmosphereOutput")
                .SetImportedResource(m_postprocTex[{ mainRtWidth, mainRtHeight }][0].get(), { 0, 0, 1, 1 });

        auto& resGbufferAlbedoMaterial =
            fg.AddResource("GbufferAlbedoMaterial")
                .SetImportedResource(perframeData.m_gbuffer.m_albedo_materialFlags.get(), { 0, 0, 1, 1 });

        auto& resGbufferNormalSmoothness =
            fg.AddResource("GbufferNormalSmoothness")
                .SetImportedResource(perframeData.m_gbuffer.m_normal_smoothness.get(), { 0, 0, 1, 1 });

        auto& resGbufferSpecularAO =
            fg.AddResource("GbufferSpecularAO")
                .SetImportedResource(perframeData.m_gbuffer.m_specular_occlusion.get(), { 0, 0, 1, 1 });

        auto& resMotionDepth =
            fg.AddResource("MotionDepth").SetImportedResource(perframeData.m_velocityMaterial.get(), { 0, 0, 1, 1 });

        auto& resDeferredShadowOutput = fg.AddResource("DeferredShadowOutput")
                                            .SetImportedResource(perframeData.m_deferShadowMask.get(), { 0, 0, 1, 1 });

        auto& resBlurredShadowIntermediateOutput =
            fg.AddResource("BlurredShadowIntermediateOutput")
                .SetImportedResource(m_postprocTex[{ mainRtWidth, mainRtHeight }][1].get(), { 0, 0, 1, 1 });

        auto& resBlurredShadowOutput = fg.AddResource("BlurredShadowOutput")
                                           .SetImportedResource(perframeData.m_deferShadowMask.get(), { 0, 0, 1, 1 });

        auto& resPrimaryViewDepth =
            fg.AddResource("PrimaryViewDepth")
                .SetImportedResource(primaryView.m_visibilityDepth_Combined.get(), { 0, 0, 1, 1 });

        auto& resDeferredShadingOutput =
            fg.AddResource("DeferredShadingOutput")
                .SetImportedResource(m_postprocTex[{ mainRtWidth, mainRtHeight }][0].get(), { 0, 0, 1, 1 });

        auto& resGlobalFogOutput =
            fg.AddResource("GlobalFogOutput").SetImportedResource(perframeData.m_taaUnresolved.get(), { 0, 0, 1, 1 });

        auto& resTAAFrameOutput =
            fg.AddResource("TAAFrameOutput")
                .SetImportedResource(m_postprocTex[{ mainRtWidth, mainRtHeight }][0].get(), { 0, 0, 1, 1 });

        auto& resTAAHistoryOutput =
            fg.AddResource("TAAHistoryOutput")
                .SetImportedResource(
                    perframeData.m_taaHistory[perframeData.m_frameId % 2].m_colorRT.get(), { 0, 0, 1, 1 });

        auto& resFinalOutput =
            fg.AddResource("FinalOutput")
                .SetImportedResource(renderTargets->GetColorAttachment(0)->GetRenderTarget(), { 0, 0, 1, 1 });

        auto& resBloomOutput =
            fg.AddResource("BloomOutput")
                .SetImportedResource(m_postprocTex[{ mainRtWidth, mainRtHeight }][1].get(), { 0, 0, 1, 1 });

        auto& resFsr2Output = fg.AddResource("Fsr2Output");

        if (m_config->m_antiAliasingType == AntiAliasingType::FSR2)
            resFsr2Output.SetImportedResource(perframeData.m_fsr2Data.m_fsr2Output.get(), { 0, 0, 1, 1 });

        // PASS START!
        auto& passAtmosphere = fg.AddPass("Atmosphere", FrameGraphPassType::Graphics)
                                   .AddReadResource(resPrimaryViewDepth)
                                   .AddWriteResource(resAtmosphereOutput);

        auto& passDeferredShadow = fg.AddPass("DeferredShadow", FrameGraphPassType::Graphics)
                                       .AddReadResource(resGbufferAlbedoMaterial)
                                       .AddReadResource(resGbufferNormalSmoothness)
                                       .AddReadResource(resGbufferSpecularAO)
                                       .AddReadResource(resMotionDepth)
                                       .AddReadResource(resPrimaryViewDepth)
                                       .AddWriteResource(resDeferredShadowOutput)
                                       .AddDependentResource(resAtmosphereOutput);
        for (auto& x : resShadowMapTexs)
        {
            passDeferredShadow.AddReadResource(*x);
        }

        auto& passBlurShadowHori = fg.AddPass("BlurShadowHori", FrameGraphPassType::Graphics)
                                       .AddReadResource(resDeferredShadowOutput)
                                       .AddWriteResource(resBlurredShadowIntermediateOutput);

        auto& passBlurShadowVert = fg.AddPass("BlurShadowVert", FrameGraphPassType::Graphics)
                                       .AddReadResource(resBlurredShadowIntermediateOutput)
                                       .AddWriteResource(resBlurredShadowOutput);

        auto& passDeferredShading = fg.AddPass("DeferredShading", FrameGraphPassType::Graphics)
                                        .AddReadResource(resBlurredShadowOutput)
                                        .AddReadResource(resGbufferAlbedoMaterial)
                                        .AddReadResource(resGbufferNormalSmoothness)
                                        .AddReadResource(resGbufferSpecularAO)
                                        .AddReadResource(resMotionDepth)
                                        .AddReadResource(resPrimaryViewDepth)
                                        .AddWriteResource(resDeferredShadingOutput)
                                        .AddDependentResource(resAtmosphereOutput);
        for (auto& x : resShadowMapTexs)
        {
            passDeferredShading.AddReadResource(*x);
        }

        // Postprocesses
        auto& passGlobalFog    = fg.AddPass("GlobalFog", FrameGraphPassType::Graphics);
        auto& passTAAResolve   = fg.AddPass("TAAResolve", FrameGraphPassType::Graphics);
        auto& passConvBloom    = fg.AddPass("ConvBloom", FrameGraphPassType::Graphics);
        auto& passFsr2Dispatch = fg.AddPass("FSR2Dispatch", FrameGraphPassType::Compute);
        auto& passToneMapping  = fg.AddPass("ToneMapping", FrameGraphPassType::Graphics);

        if (m_config->m_antiAliasingType == AntiAliasingType::FSR2)
        {
            passGlobalFog.AddReadResource(resPrimaryViewDepth)
                .AddReadResource(resDeferredShadingOutput)
                .AddWriteResource(resGlobalFogOutput);

            passConvBloom.AddReadResource(resGlobalFogOutput).AddWriteResource(resBloomOutput);

            passFsr2Dispatch.AddReadResource(resBloomOutput).AddWriteResource(resFsr2Output);
        }
        else if (m_config->m_antiAliasingType == AntiAliasingType::TAA)
        {
            passGlobalFog.AddReadResource(resPrimaryViewDepth)
                .AddReadResource(resDeferredShadingOutput)
                .AddWriteResource(resGlobalFogOutput);

            passTAAResolve.AddReadResource(resGlobalFogOutput)
                .AddWriteResource(resTAAFrameOutput)
                .AddWriteResource(resTAAHistoryOutput);

            passConvBloom.AddReadResource(resTAAFrameOutput).AddWriteResource(resBloomOutput);
        }
        else
        {
            passGlobalFog.AddReadResource(resPrimaryViewDepth)
                .AddReadResource(resDeferredShadingOutput)
                .AddWriteResource(resGlobalFogOutput);

            passConvBloom.AddReadResource(resGlobalFogOutput).AddWriteResource(resBloomOutput);
        }

        // Tonemapping
        if (m_config->m_antiAliasingType == AntiAliasingType::FSR2)
        {
            passToneMapping.AddReadResource(resFsr2Output).AddWriteResource(resFinalOutput);
        }
        else
        {
            passToneMapping.AddReadResource(resBloomOutput).AddWriteResource(resFinalOutput);
        }

        // Pass Exec
        passAtmosphere.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto commandList = data.m_CmdList;
            auto postprocId  = m_postprocTex[{ mainRtWidth, mainRtHeight }][0]->GetDescId();
            auto postprocTex = m_postprocTex[{ mainRtWidth, mainRtHeight }][0];
            auto atmoData    = m_atmosphereRenderer->GetResourceDesc(perframeData);
            auto rhi         = m_app->GetRhi();
            struct AtmoPushConst
            {
                Vector4f           sundir;
                u32                m_perframe;
                u32                m_outTex;
                u32                m_depthTex;
                u32                pad1;
                decltype(atmoData) m_atmoData;
            } pushConst;
            pushConst.sundir     = perframeData.m_sunDir;
            pushConst.m_perframe = primaryView.m_viewBufferId->GetActiveId();
            pushConst.m_outTex   = postprocId;
            pushConst.m_atmoData = atmoData;
            pushConst.m_depthTex = primaryView.m_visibilityDepthIdSRV_Combined->GetActiveId();
            m_atmospherePass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
                ctx->m_cmd->SetPushConst(m_atmospherePass, 0, sizeof(AtmoPushConst), &pushConst);
                auto wgX = Math::DivRoundUp(primaryView.m_renderWidth, SyaroConfig::cAtmoRenderThreadGroupSizeX);
                auto wgY = Math::DivRoundUp(primaryView.m_renderHeight, SyaroConfig::cAtmoRenderThreadGroupSizeY);
                ctx->m_cmd->Dispatch(wgX, wgY, 1);
            });
            commandList->BeginScope("Syaro: Atmosphere");
            m_atmospherePass->Run(commandList, 0);
            commandList->EndScope();
        });

        passDeferredShadow.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto commandList = data.m_CmdList;
            struct DeferPushConst
            {
                u32 shadowMapDataRef;
                u32 numShadowMaps;
                u32 depthTexRef;
            } pc;
            pc.numShadowMaps    = perframeData.m_shadowData2.m_enabledShadowMaps;
            pc.shadowMapDataRef = perframeData.m_shadowData2.m_allShadowDataId->GetActiveId();
            pc.depthTexRef      = primaryView.m_visibilityDepthIdSRV_Combined->GetActiveId();
            commandList->BeginScope("Syaro: Deferred Shadowing");

            auto targetRT = perframeData.m_deferShadowMaskRTs.get();
            RenderingUtil::EnqueueFullScreenPass(commandList, rhi, m_deferredShadowPass, targetRT,
                { perframeData.m_gbufferDescFrag, perframeData.m_gbufferDepthDesc, primaryView.m_viewBindlessRef }, &pc,
                3);
            commandList->EndScope();
        });

        passBlurShadowHori.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto postprocRTs   = m_postprocRTs[{ mainRtWidth, mainRtHeight }];
            auto postprocRT1   = postprocRTs[1];
            auto deferShadowId = perframeData.m_deferShadowMaskId.get();
            m_gaussianHori->RenderPostFx(data.m_CmdList, postprocRT1.get(), deferShadowId, 3);
        });

        passBlurShadowVert.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto postprocId = m_postprocTexSRV[{ mainRtWidth, mainRtHeight }][1];
            m_gaussianVert->RenderPostFx(data.m_CmdList, perframeData.m_deferShadowMaskRTs.get(), postprocId.get(), 3);
        });

        passDeferredShading.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto commandList = data.m_CmdList;
            SetupDeferredShadingPass(renderTargets);
            auto                      postprocRTs = m_postprocRTs[{ mainRtWidth, mainRtHeight }];
            auto                      postprocRT0 = postprocRTs[0];
            auto                      curRT       = perframeData.m_taaHistory[perframeData.m_frameId % 2].m_rts;
            PipelineAttachmentConfigs paCfg;
            auto                      rtCfg = curRT->GetFormat();
            paCfg.m_colorFormats            = rtCfg.m_colorFormats;
            paCfg.m_depthFormat             = RhiImageFormat::RhiImgFmt_UNDEFINED;
            auto pass                       = m_deferredShadingPass[paCfg];
            struct DeferPushConst
            {
                Vector4f sundir;
                u32      shadowMapDataRef;
                u32      numShadowMaps;
                u32      depthTexRef;
                u32      shadowTexRef;
            } pc;
            pc.sundir           = perframeData.m_sunDir;
            pc.numShadowMaps    = perframeData.m_shadowData2.m_enabledShadowMaps;
            pc.shadowMapDataRef = perframeData.m_shadowData2.m_allShadowDataId->GetActiveId();
            pc.depthTexRef      = primaryView.m_visibilityDepthIdSRV_Combined->GetActiveId();
            pc.shadowTexRef     = perframeData.m_deferShadowMaskId->GetActiveId();
            commandList->BeginScope("Syaro: Deferred Shading");
            RenderingUtil::EnqueueFullScreenPass(commandList, rhi, pass, postprocRT0.get(),
                { perframeData.m_gbufferDescFrag, perframeData.m_gbufferDepthDesc, primaryView.m_viewBindlessRef }, &pc,
                8);
            commandList->EndScope();
        });

        passGlobalFog.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto fogRT         = perframeData.m_taaHistory[perframeData.m_frameId % 2].m_rts;
            auto inputId       = m_postprocTexSRV[{ mainRtWidth, mainRtHeight }][0].get();
            auto inputDepthId  = primaryView.m_visibilityDepthIdSRV_Combined.get();
            auto primaryViewId = primaryView.m_viewBufferId.get();
            m_globalFogPass->RenderPostFx(data.m_CmdList, fogRT.get(), inputId, inputDepthId, primaryViewId);
        });

        if (m_config->m_antiAliasingType == AntiAliasingType::TAA)
        {
            passTAAResolve.SetExecutionFunction([&](const FrameGraphPassContext& data) {
                auto commandList = data.m_CmdList;
                auto taaRT       = rhi->CreateRenderTargets();
                auto taaCurTarget =
                    rhi->CreateRenderTarget(perframeData.m_taaHistory[perframeData.m_frameId % 2].m_colorRT.get(), {},
                        RhiRenderTargetLoadOp::Clear, 0, 0);
                auto taaRenderTarget = m_postprocColorRT[{ mainRtWidth, mainRtHeight }][0].get();

                taaRT->SetColorAttachments({ taaCurTarget.get(), taaRenderTarget });
                taaRT->SetDepthStencilAttachment(nullptr);
                taaRT->SetRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));
                SetupTAAPass(renderTargets);
                auto                      rtCfg = taaRT->GetFormat();
                PipelineAttachmentConfigs paCfg;
                paCfg.m_colorFormats = rtCfg.m_colorFormats;
                paCfg.m_depthFormat  = RhiImageFormat::RhiImgFmt_UNDEFINED;
                auto pass            = m_taaPass[paCfg];
                u32  jitterX         = std::bit_cast<u32, float>(perframeData.m_taaJitterX);
                u32  jitterY         = std::bit_cast<u32, float>(perframeData.m_taaJitterY);
                u32  pconst[5]       = {
                    perframeData.m_frameId,
                    mainRtWidth,
                    mainRtHeight,
                    jitterX,
                    jitterY,
                };
                commandList->BeginScope("Syaro: TAA Resolve");
                RenderingUtil::EnqueueFullScreenPass(commandList, rhi, pass, taaRT.get(),
                    { perframeData.m_taaHistoryDesc, perframeData.m_gbufferDepthDesc }, pconst, 3);
                commandList->EndScope();
            });

            passConvBloom.SetExecutionFunction([&](const FrameGraphPassContext& data) {
                auto commandList = data.m_CmdList;
                commandList->BeginScope("Syaro: Convolution Bloom");
                auto width          = mainRtWidth;
                auto height         = mainRtHeight;
                auto postprocTex0Id = m_postprocTexSRV[{ width, height }][0].get();
                auto postprocTex1Id = m_postprocTex[{ width, height }][1]->GetDescId();
                m_fftConv2d->RenderPostFx(
                    commandList, postprocTex0Id, postprocTex1Id, nullptr, width, height, 51, 51, 4);
                commandList->EndScope();
            });
        }
        else
        {
            passTAAResolve.SetExecutionFunction([&](const FrameGraphPassContext& data) {});
            passConvBloom.SetExecutionFunction([&](const FrameGraphPassContext& data) {
                auto commandList = data.m_CmdList;
                commandList->BeginScope("Syaro: Convolution Bloom");
                auto width          = mainRtWidth;
                auto height         = mainRtHeight;
                auto postprocTex0Id = perframeData.m_taaHistory[perframeData.m_frameId % 2].m_colorRTIdSRV.get();
                auto postprocTex1Id = m_postprocTex[{ width, height }][1]->GetDescId();
                m_fftConv2d->RenderPostFx(
                    commandList, postprocTex0Id, postprocTex1Id, nullptr, width, height, 51, 51, 4);
                commandList->EndScope();
            });
        }

        if (m_config->m_antiAliasingType == AntiAliasingType::FSR2)
        {
            passFsr2Dispatch.SetExecutionFunction([&](const FrameGraphPassContext& data) {
                auto commandList = data.m_CmdList;
                commandList->BeginScope("Syaro: FSR2 Dispatch");
                auto                                     color  = m_postprocTex[{ mainRtWidth, mainRtHeight }][0].get();
                auto                                     depth  = primaryView.m_visibilityDepth_Combined.get();
                auto                                     motion = perframeData.m_motionVector.get();
                auto                                     mainView = GetPrimaryView(perframeData);
                Graphics::Rhi::FSR2::RhiFSR2DispatchArgs args;
                args.camFar  = mainView.m_viewData.m_cameraFar;
                args.camNear = mainView.m_viewData.m_cameraNear;
                args.camFovY = mainView.m_viewData.m_cameraFovY;
                args.color   = color;
                args.depth   = depth;
                if (perframeData.m_frameId < 2)
                {
                    args.deltaTime = 16.6f;
                }
                else
                {
                    // TODO: precision loss
                    args.deltaTime = perframeData.m_frameTimestamp[perframeData.m_frameId % 2]
                        - perframeData.m_frameTimestamp[(perframeData.m_frameId - 1) % 2];
                    args.deltaTime = std::max(args.deltaTime, 16.6f);
                }
                args.exposure         = nullptr;
                args.jitterX          = perframeData.m_taaJitterX;
                args.jitterY          = perframeData.m_taaJitterY;
                args.motion           = motion;
                args.reactiveMask     = nullptr;
                args.transparencyMask = nullptr;
                args.output           = perframeData.m_fsr2Data.m_fsr2Output.get();

                args.reset = primaryView.m_camMoved;

                if (args.output == nullptr)
                {
                    throw std::runtime_error("FSR2 output is null");
                }

                commandList->BeginScope("Syaro: FSR2 Dispatch, Impl");
                m_fsr2proc->Dispatch(commandList, args);
                commandList->EndScope();
                commandList->EndScope();
            });
        }
        else
        {
            passFsr2Dispatch.SetExecutionFunction([&](const FrameGraphPassContext& data) {});
        }

        if (m_config->m_antiAliasingType == AntiAliasingType::FSR2)
        {
            auto renderArea = renderTargets->GetRenderArea();
            auto width      = renderArea.width;
            auto height     = renderArea.height;

            passToneMapping.SetExecutionFunction([&](const FrameGraphPassContext& data) {
                m_acesToneMapping->RenderPostFx(
                    data.m_CmdList, renderTargets, perframeData.m_fsr2Data.m_fsr2OutputSRVId.get());
            });
        }
        else
        {
            passToneMapping.SetExecutionFunction([&](const FrameGraphPassContext& data) {
                m_acesToneMapping->RenderPostFx(
                    data.m_CmdList, renderTargets, m_postprocTexSRV[{ mainRtWidth, mainRtHeight }][0].get());
            });
        }

        // transition input resources
        if (m_config->m_antiAliasingType == AntiAliasingType::FSR2)
        {
            runImageBarrier(cmd, perframeData.m_fsr2Data.m_fsr2Output.get(), RhiResourceState::Common, { 0, 0, 1, 1 });

            auto motion      = perframeData.m_motionVector.get();
            auto primaryView = GetPrimaryView(perframeData);
            auto depth       = primaryView.m_visibilityDepth_Combined.get();

            runImageBarrier(cmd, motion, RhiResourceState::ShaderRead, { 0, 0, 1, 1 });
            runImageBarrier(cmd, depth, RhiResourceState::ShaderRead, { 0, 0, 1, 1 });
        }
        runImageBarrier(
            cmd, m_postprocTex[{ mainRtWidth, mainRtHeight }][0].get(), RhiResourceState::ColorRT, { 0, 0, 1, 1 });
        runImageBarrier(
            cmd, m_postprocTex[{ mainRtWidth, mainRtHeight }][1].get(), RhiResourceState::ColorRT, { 0, 0, 1, 1 });

        // run!
        FrameGraphCompiler compiler;
        auto               compiledGraph = compiler.Compile(fg);
        FrameGraphExecutor executor;

        cmd->BeginScope("Syaro: Deferred Shading");
        executor.ExecuteInSingleCmd(cmd, compiledGraph);
        cmd->EndScope();
    }

    IFRIT_APIDECL void SyaroRenderer::SetupPbrAtmosphereRenderer()
    {
        m_atmosphereRenderer = std::make_shared<PbrAtmosphereRenderer>(m_app);
        auto rhi             = m_app->GetRhi();
        m_atmospherePass     = RenderingUtil::CreateComputePass(rhi, "Syaro/Syaro.PbrAtmoRender.comp.glsl", 0, 19);
    }

    IFRIT_APIDECL void SyaroRenderer::SetupDeferredShadingPass(RenderTargets* renderTargets)
    {
        auto                      rhi = m_app->GetRhi();

        // This seems to be a bit of redundant code
        // The rhi backend now can reference the pipeline with similar
        // CI

        PipelineAttachmentConfigs paCfg;
        auto                      rtCfg = renderTargets->GetFormat();
        paCfg.m_colorFormats            = { cTAAFormat };
        paCfg.m_depthFormat             = RhiImageFormat::RhiImgFmt_UNDEFINED;

        rtCfg.m_colorFormats = paCfg.m_colorFormats;
        rtCfg.m_depthFormat  = RhiImageFormat::RhiImgFmt_UNDEFINED;

        DrawPass* pass = nullptr;
        if (m_deferredShadingPass.find(paCfg) != m_deferredShadingPass.end())
        {
            pass = m_deferredShadingPass[paCfg];
        }
        else
        {
            pass          = rhi->CreateGraphicsPass();
            auto vsShader = CreateShaderFromFile("Syaro.DeferredShading.vert.glsl", "main", RhiShaderStage::Vertex);
            auto fsShader = CreateShaderFromFile("Syaro.DeferredShading.frag.glsl", "main", RhiShaderStage::Fragment);
            pass->SetVertexShader(vsShader);
            pass->SetPixelShader(fsShader);
            pass->SetNumBindlessDescriptorSets(3);
            pass->SetPushConstSize(8 * u32Size);
            pass->SetRenderTargetFormat(rtCfg);
            m_deferredShadingPass[paCfg] = pass;
        }
    }

    IFRIT_APIDECL void SyaroRenderer::SetupDebugPasses(PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        auto                      rhi   = m_app->GetRhi();
        auto                      rtCfg = renderTargets->GetFormat();

        PipelineAttachmentConfigs paCfg;
        paCfg.m_colorFormats = rtCfg.m_colorFormats;
        paCfg.m_depthFormat  = rtCfg.m_depthFormat;

        if (m_triangleViewPass.find(paCfg) == m_triangleViewPass.end())
        {
            auto pass     = rhi->CreateGraphicsPass();
            auto vsShader = CreateShaderFromFile("Syaro.TriangleView.vert.glsl", "main", RhiShaderStage::Vertex);
            auto fsShader = CreateShaderFromFile("Syaro.TriangleView.frag.glsl", "main", RhiShaderStage::Fragment);
            pass->SetVertexShader(vsShader);
            pass->SetPixelShader(fsShader);
            pass->SetNumBindlessDescriptorSets(0);
            pass->SetPushConstSize(u32Size);
            pass->SetRenderTargetFormat(rtCfg);

            m_triangleViewPass[paCfg] = pass;
        }
    }

    IFRIT_APIDECL void SyaroRenderer::SetupTAAPass(RenderTargets* renderTargets)
    {
        auto                      rhi = m_app->GetRhi();

        PipelineAttachmentConfigs paCfg;
        auto                      rtCfg = renderTargets->GetFormat();
        paCfg.m_colorFormats            = { cTAAFormat, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT };
        paCfg.m_depthFormat             = RhiImageFormat::RhiImgFmt_UNDEFINED;
        rtCfg.m_colorFormats            = paCfg.m_colorFormats;

        DrawPass* pass = nullptr;
        if (m_taaPass.find(paCfg) != m_taaPass.end())
        {
            pass = m_taaPass[paCfg];
        }
        else
        {
            pass          = rhi->CreateGraphicsPass();
            auto vsShader = CreateShaderFromFile("Syaro.TAA.vert.glsl", "main", RhiShaderStage::Vertex);
            auto fsShader = CreateShaderFromFile("Syaro.TAA.frag.glsl", "main", RhiShaderStage::Fragment);
            pass->SetVertexShader(vsShader);
            pass->SetPixelShader(fsShader);
            pass->SetNumBindlessDescriptorSets(2);
            pass->SetPushConstSize(u32Size * 5);
            rtCfg.m_depthFormat = RhiImageFormat::RhiImgFmt_UNDEFINED;
            paCfg.m_depthFormat = RhiImageFormat::RhiImgFmt_UNDEFINED;
            pass->SetRenderTargetFormat(rtCfg);
            m_taaPass[paCfg] = pass;
        }
    }

    IFRIT_APIDECL void SyaroRenderer::SetupVisibilityPass()
    {
        auto rhi = m_app->GetRhi();

        // Hardware Rasterize
        if IF_CONSTEXPR (true)
        {
            auto tsShader      = CreateShaderFromFile("Syaro.VisBuffer.task.glsl", "main", RhiShaderStage::Task);
            auto msShader      = CreateShaderFromFile("Syaro.VisBuffer.mesh.glsl", "main", RhiShaderStage::Mesh);
            auto msShaderDepth = CreateShaderFromFile("Syaro.VisBufferDepth.mesh.glsl", "main", RhiShaderStage::Mesh);
            auto fsShader      = CreateShaderFromFile("Syaro.VisBuffer.frag.glsl", "main", RhiShaderStage::Fragment);

            m_visibilityPassHW = rhi->CreateGraphicsPass();
#if !SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
            m_visibilityPassHW->SetTaskShader(tsShader);
#endif
            m_visibilityPassHW->SetMeshShader(msShader);
            m_visibilityPassHW->SetPixelShader(fsShader);
            m_visibilityPassHW->SetNumBindlessDescriptorSets(3);
            m_visibilityPassHW->SetPushConstSize(u32Size);

            RhiRenderTargetsFormat rtFmt;
            rtFmt.m_colorFormats = { RhiImageFormat::RhiImgFmt_R32_UINT };
            rtFmt.m_depthFormat  = RhiImageFormat::RhiImgFmt_D32_SFLOAT;
            m_visibilityPassHW->SetRenderTargetFormat(rtFmt);

            m_depthOnlyVisibilityPassHW = rhi->CreateGraphicsPass();
#if !SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
            m_depthOnlyVisibilityPassHW->SetTaskShader(tsShader);
#endif
            m_depthOnlyVisibilityPassHW->SetMeshShader(msShaderDepth);
            m_depthOnlyVisibilityPassHW->SetNumBindlessDescriptorSets(3);
            m_depthOnlyVisibilityPassHW->SetPushConstSize(u32Size);

            RhiRenderTargetsFormat depthOnlyRtFmt;
            depthOnlyRtFmt.m_colorFormats = {};
            depthOnlyRtFmt.m_depthFormat  = RhiImageFormat::RhiImgFmt_D32_SFLOAT;
            m_depthOnlyVisibilityPassHW->SetRenderTargetFormat(depthOnlyRtFmt);
        }

#if SYARO_ENABLE_SW_RASTERIZER
        // Software Rasterize
        if (true)
        {
            auto csShader      = CreateShaderFromFile("Syaro.SoftRasterize.comp.glsl", "main", RhiShaderStage::Compute);
            m_visibilityPassSW = rhi->CreateComputePass();
            m_visibilityPassSW->SetComputeShader(csShader);
            m_visibilityPassSW->SetNumBindlessDescriptorSets(3);
            m_visibilityPassSW->SetPushConstSize(u32Size * 8);
        }

        // Combine software and hardware rasterize results
        if (true)
        {
            auto csShader = CreateShaderFromFile("Syaro.CombineVisBuffer.comp.glsl", "main", RhiShaderStage::Compute);
            m_visibilityCombinePass = rhi->CreateComputePass();
            m_visibilityCombinePass->SetComputeShader(csShader);
            m_visibilityCombinePass->SetNumBindlessDescriptorSets(0);
            m_visibilityCombinePass->SetPushConstSize(u32Size * 9);
        }
#endif
    }

    IFRIT_APIDECL void SyaroRenderer::SetupInstanceCullingPass()
    {
        auto rhi              = m_app->GetRhi();
        m_instanceCullingPass = RenderingUtil::CreateComputePass(rhi, "Syaro/Syaro.InstanceCulling.comp.glsl", 4, 2);
    }

    IFRIT_APIDECL void SyaroRenderer::SetupPersistentCullingPass()
    {
        auto rhi = m_app->GetRhi();
        m_persistentCullingPass =
            RenderingUtil::CreateComputePass(rhi, "Syaro/Syaro.PersistentCulling.comp.glsl", 5, 3);

        m_indirectDrawBuffer = rhi->CreateBufferDevice("Syaro_IndirectDraw", u32Size * 1, kbBufUsage_Indirect, true);
        m_persistCullDesc    = rhi->createBindlessDescriptorRef();
        m_persistCullDesc->AddStorageBuffer(m_indirectDrawBuffer.get(), 0);
    }

    IFRIT_APIDECL void SyaroRenderer::SetupSinglePassHiZPass()
    {
        m_singlePassHiZProc = std::make_shared<SinglePassHiZPass>(m_app);
    }
    IFRIT_APIDECL void SyaroRenderer::SetupEmitDepthTargetsPass()
    {
        auto rhi               = m_app->GetRhi();
        m_emitDepthTargetsPass = RenderingUtil::CreateComputePass(rhi, "Syaro/Syaro.EmitDepthTarget.comp.glsl", 4, 2);
    }

    IFRIT_APIDECL void SyaroRenderer::SetupMaterialClassifyPass()
    {
        auto rhi = m_app->GetRhi();
        m_matclassCountPass =
            RenderingUtil::CreateComputePass(rhi, "Syaro/Syaro.ClassifyMaterial.Count.comp.glsl", 1, 3);
        m_matclassReservePass =
            RenderingUtil::CreateComputePass(rhi, "Syaro/Syaro.ClassifyMaterial.Reserve.comp.glsl", 1, 3);
        m_matclassScatterPass =
            RenderingUtil::CreateComputePass(rhi, "Syaro/Syaro.ClassifyMaterial.Scatter.comp.glsl", 1, 3);
    }

    IFRIT_APIDECL void SyaroRenderer::MaterialClassifyBufferSetup(
        PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        auto numMaterials = perframeData.m_enabledEffects.size();
        auto rhi          = m_app->GetRhi();

        u32  actualRtWidth = 0, actualRtHeight = 0;
        GetSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

        auto width             = actualRtWidth;
        auto height            = actualRtHeight;
        auto totalSize         = width * height;
        bool needRecreate      = false;
        bool needRecreateMat   = false;
        bool needRecreatePixel = false;
        if (perframeData.m_matClassSupportedNumMaterials < numMaterials
            || perframeData.m_matClassCountBuffer == nullptr)
        {
            needRecreate    = true;
            needRecreateMat = true;
        }
        if (perframeData.m_matClassSupportedNumPixels < totalSize)
        {
            needRecreate      = true;
            needRecreatePixel = true;
        }
        if (!needRecreate)
        {
            return;
        }
        if (needRecreateMat)
        {
            perframeData.m_matClassSupportedNumMaterials = Ifrit::Common::Utility::SizeCast<u32>(numMaterials);
            auto CreateSize                              = cMatClassCounterBufferSizeBase
                + cMatClassCounterBufferSizeMult * Ifrit::Common::Utility::SizeCast<u32>(numMaterials);

            perframeData.m_matClassCountBuffer =
                rhi->CreateBufferDevice("Syaro_MatClassCount", CreateSize, kbBufUsage_SSBO_CopyDest, true);
            perframeData.m_matClassIndirectDispatchBuffer = rhi->CreateBufferDevice(
                "Syaro_MatClassCountIndirectDisp", u32Size * 4 * numMaterials, kbBufUsage_Indirect, true);
        }
        if (needRecreatePixel)
        {
            perframeData.m_matClassSupportedNumPixels = totalSize;
            perframeData.m_matClassFinalBuffer =
                rhi->CreateBufferDevice("Syaro_MatClassFinal", u32Size * totalSize, kbBufUsage_SSBO, true);
            perframeData.m_matClassPixelOffsetBuffer = rhi->CreateBufferDevice(
                "Syaro_MatClassPixelOffset", u32Size * totalSize, kbBufUsage_SSBO_CopyDest, true);
        }

        if (needRecreate)
        {
            perframeData.m_matClassDesc = rhi->createBindlessDescriptorRef();
            perframeData.m_matClassDesc->AddUAVImage(perframeData.m_velocityMaterial.get(), { 0, 0, 1, 1 }, 0);
            perframeData.m_matClassDesc->AddStorageBuffer(perframeData.m_matClassCountBuffer.get(), 1);
            perframeData.m_matClassDesc->AddStorageBuffer(perframeData.m_matClassFinalBuffer.get(), 2);
            perframeData.m_matClassDesc->AddStorageBuffer(perframeData.m_matClassPixelOffsetBuffer.get(), 3);
            perframeData.m_matClassDesc->AddStorageBuffer(perframeData.m_matClassIndirectDispatchBuffer.get(), 4);

            perframeData.m_matClassBarrier.clear();
            RhiResourceBarrier barrierCountBuffer;
            barrierCountBuffer.m_type         = RhiBarrierType::UAVAccess;
            barrierCountBuffer.m_uav.m_buffer = perframeData.m_matClassCountBuffer.get();
            barrierCountBuffer.m_uav.m_type   = RhiResourceType::Buffer;

            RhiResourceBarrier barrierFinalBuffer;
            barrierFinalBuffer.m_type         = RhiBarrierType::UAVAccess;
            barrierFinalBuffer.m_uav.m_buffer = perframeData.m_matClassFinalBuffer.get();
            barrierFinalBuffer.m_uav.m_type   = RhiResourceType::Buffer;

            RhiResourceBarrier barrierPixelOffsetBuffer;
            barrierPixelOffsetBuffer.m_type         = RhiBarrierType::UAVAccess;
            barrierPixelOffsetBuffer.m_uav.m_buffer = perframeData.m_matClassPixelOffsetBuffer.get();
            barrierPixelOffsetBuffer.m_uav.m_type   = RhiResourceType::Buffer;

            RhiResourceBarrier barrierIndirectDispatchBuffer;
            barrierIndirectDispatchBuffer.m_type         = RhiBarrierType::UAVAccess;
            barrierIndirectDispatchBuffer.m_uav.m_buffer = perframeData.m_matClassIndirectDispatchBuffer.get();
            barrierIndirectDispatchBuffer.m_uav.m_type   = RhiResourceType::Buffer;

            perframeData.m_matClassBarrier.push_back(barrierCountBuffer);
            perframeData.m_matClassBarrier.push_back(barrierFinalBuffer);
            perframeData.m_matClassBarrier.push_back(barrierPixelOffsetBuffer);
            perframeData.m_matClassBarrier.push_back(barrierIndirectDispatchBuffer);
        }
    }

    IFRIT_APIDECL void SyaroRenderer::DepthTargetsSetup(PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        auto rhi = m_app->GetRhi();

        u32  actualRtWidth = 0, actualRtHeight = 0;
        GetSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

        if (perframeData.m_velocityMaterial != nullptr)
            return;
        perframeData.m_velocityMaterial = rhi->CreateTexture2D(
            "Syaro_VelocityMat", actualRtWidth, actualRtHeight, kbImFmt_RGBA32F, kbImUsage_UAV_SRV, true);
        perframeData.m_motionVector = rhi->CreateTexture2D(
            "Syaro_Motion", actualRtWidth, actualRtHeight, kbImFmt_RG32F, kbImUsage_UAV_SRV_CopyDest, true);

        perframeData.m_velocityMaterialDesc = rhi->createBindlessDescriptorRef();
        perframeData.m_velocityMaterialDesc->AddUAVImage(perframeData.m_velocityMaterial.get(), { 0, 0, 1, 1 }, 0);

        auto& primaryView = GetPrimaryView(perframeData);
        perframeData.m_velocityMaterialDesc->AddCombinedImageSampler(
            primaryView.m_visibilityBuffer_Combined.get(), m_immRes.m_linearSampler.get(), 1);
        perframeData.m_velocityMaterialDesc->AddUAVImage(perframeData.m_motionVector.get(), { 0, 0, 1, 1 }, 2);

        // For gbuffer, depth is required to reconstruct position
        perframeData.m_gbufferDepthDesc = rhi->createBindlessDescriptorRef();
        perframeData.m_gbufferDepthDesc->AddCombinedImageSampler(
            perframeData.m_velocityMaterial.get(), m_immRes.m_linearSampler.get(), 0);
    }

    IFRIT_APIDECL void SyaroRenderer::RecreateInstanceCullingBuffers(PerFrameData& perframe, u32 newMaxInstances)
    {
        for (u32 i = 0; i < perframe.m_views.size(); i++)
        {
            auto& view = perframe.m_views[i];
            if (view.m_maxSupportedInstances == 0 || view.m_maxSupportedInstances < newMaxInstances)
            {
                auto rhi                     = m_app->GetRhi();
                view.m_maxSupportedInstances = newMaxInstances;
                view.m_instCullDiscardObj =
                    rhi->CreateBufferDevice("Syaro_InstCullDiscard", u32Size * newMaxInstances, kbBufUsage_SSBO, true);
                view.m_instCullPassedObj = rhi->CreateBufferDevice(
                    "Syaro_InstCullPassed", u32Size * newMaxInstances, kbBufUsage_SSBO_CopyDest, true);
                view.m_persistCullIndirectDispatch =
                    rhi->CreateBufferDevice("Syaro_InstCullDispatch", u32Size * 12, kbBufUsage_Indirect, true);

                view.m_instCullDesc = rhi->createBindlessDescriptorRef();
                view.m_instCullDesc->AddStorageBuffer(view.m_instCullDiscardObj.get(), 0);
                view.m_instCullDesc->AddStorageBuffer(view.m_instCullPassedObj.get(), 1);
                view.m_instCullDesc->AddStorageBuffer(view.m_persistCullIndirectDispatch.get(), 2);

                // create barriers
                view.m_persistCullBarrier.clear();
                view.m_persistCullBarrier =
                    RegisterUAVBarriers({ view.m_instCullDiscardObj.get(), view.m_instCullPassedObj.get(),
                                            view.m_persistCullIndirectDispatch.get() },
                        {});

                view.m_visibilityBarrier.clear();
                view.m_visibilityBarrier =
                    RegisterUAVBarriers({ view.m_allFilteredMeshletsHW.get(), view.m_allFilteredMeshletsAllCount.get(),
                                            view.m_allFilteredMeshletsSW.get() },
                        {});
            }
        }
    }

    IFRIT_APIDECL void SyaroRenderer::RenderEmitDepthTargets(
        PerFrameData& perframeData, SyaroRenderer::RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        auto  rhi         = m_app->GetRhi();
        auto& primaryView = GetPrimaryView(perframeData);
        m_emitDepthTargetsPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            auto primaryView = GetPrimaryView(perframeData);
#if !SYARO_ENABLE_SW_RASTERIZER
            // This barrier is intentionally for layout transition. Sync barrier is
            // issued via GlobalMemoryBarrier
            ctx->m_cmd->AddImageBarrier(primaryView.m_visibilityDepth_Combined.get(),
                RhiResourceState::DepthStencilRenderTarget, RhiResourceState::Common, { 0, 0, 1, 1 });
#endif
            runImageBarrier(ctx->m_cmd, perframeData.m_velocityMaterial.get(),

                RhiResourceState::UnorderedAccess, { 0, 0, 1, 1 });
            runImageBarrier(ctx->m_cmd, perframeData.m_motionVector.get(), RhiResourceState::Common, { 0, 0, 1, 1 });
            ctx->m_cmd->ClearUAVTexFloat(perframeData.m_motionVector.get(), { 0, 0, 1, 1 }, { 0.0f, 0.0f, 0.0f, 0.0f });
            ctx->m_cmd->AttachBindlessRefCompute(m_emitDepthTargetsPass, 1, primaryView.m_viewBindlessRef);
            ctx->m_cmd->AttachBindlessRefCompute(
                m_emitDepthTargetsPass, 2, perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
            ctx->m_cmd->AttachBindlessRefCompute(m_emitDepthTargetsPass, 3, primaryView.m_allFilteredMeshletsDesc);
            ctx->m_cmd->AttachBindlessRefCompute(m_emitDepthTargetsPass, 4, perframeData.m_velocityMaterialDesc);
            u32 pcData[2] = { primaryView.m_renderWidth, primaryView.m_renderHeight };
            ctx->m_cmd->SetPushConst(m_emitDepthTargetsPass, 0, u32Size * 2, &pcData[0]);
            u32 wgX = (pcData[0] + cEmitDepthGroupSizeX - 1) / cEmitDepthGroupSizeX;
            u32 wgY = (pcData[1] + cEmitDepthGroupSizeY - 1) / cEmitDepthGroupSizeY;
            ctx->m_cmd->Dispatch(wgX, wgY, 1);

            runImageBarrier(ctx->m_cmd, perframeData.m_velocityMaterial.get(),

                RhiResourceState::UnorderedAccess, { 0, 0, 1, 1 });

#if !SYARO_ENABLE_SW_RASTERIZER
            runImageBarrier(ctx->m_cmd, primaryView.m_visibilityDepth_Combined.get(), RhiResourceState::DepthStencilRT,
                { 0, 0, 1, 1 });
#endif
        });

        cmd->BeginScope("Syaro: Emit Depth Targets");
        m_emitDepthTargetsPass->Run(cmd, 0);
        cmd->EndScope();
    }
    IFRIT_APIDECL void SyaroRenderer::RenderTwoPassOcclCulling(CullingPass cullPass, PerFrameData& perframeData,
        RenderTargets* renderTargets, const GPUCmdBuffer* cmd, PerFrameData::ViewType filteredViewType, u32 idx)
    {
        auto                                                 rhi       = m_app->GetRhi();
        int                                                  pcData[2] = { 0, 1 };

        std::unique_ptr<SyaroRenderer::GPUCommandSubmission> lastTask = nullptr;
        u32                                                  k        = idx;
        if (k == ~0u)
        {
            for (k = 0; k < perframeData.m_views.size(); k++)
            {
                if (filteredViewType == perframeData.m_views[k].m_viewType)
                {
                    break;
                }
            }
        }
        if (filteredViewType != perframeData.m_views[k].m_viewType)
        {
            return;
        }
        if (k != 0 && cullPass != CullingPass::First)
        {
            cmd->GlobalMemoryBarrier();
        }
        auto& perView           = perframeData.m_views[k];
        auto  numObjs           = perframeData.m_allInstanceData.m_objectData.size();
        int   pcDataInstCull[4] = { 0, Ifrit::Common::Utility::SizeCast<int>(numObjs), 1,
              Ifrit::Common::Utility::SizeCast<int>(numObjs) };
        m_instanceCullingPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            if (cullPass == CullingPass::First)
            {
                ctx->m_cmd->BufferClear(perView.m_persistCullIndirectDispatch.get(), 0);
            }
            runUAVBufferBarrier(ctx->m_cmd, perView.m_persistCullIndirectDispatch.get());
            ctx->m_cmd->AttachBindlessRefCompute(m_instanceCullingPass, 1, perView.m_viewBindlessRef);
            ctx->m_cmd->AttachBindlessRefCompute(
                m_instanceCullingPass, 2, perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
            ctx->m_cmd->AttachBindlessRefCompute(m_instanceCullingPass, 3, perView.m_instCullDesc);
            ctx->m_cmd->AttachBindlessRefCompute(m_instanceCullingPass, 4, perView.m_spHiZData.m_hizDesc);

            if (cullPass == CullingPass::First)
            {
                ctx->m_cmd->SetPushConst(m_instanceCullingPass, 0, u32Size * 2, &pcDataInstCull[0]);
                auto tgx = DivRoundUp(SizeCast<u32>(numObjs), SyaroConfig::cInstanceCullingThreadGroupSizeX);
                ctx->m_cmd->Dispatch(tgx, 1, 1);
            }
            else if (cullPass == CullingPass::Second)
            {
                ctx->m_cmd->SetPushConst(m_instanceCullingPass, 0, u32Size * 2, &pcDataInstCull[2]);
                ctx->m_cmd->DispatchIndirect(perView.m_persistCullIndirectDispatch.get(), 3 * u32Size);
            }
        });

        m_persistentCullingPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            struct PersistCullPushConst
            {
                u32 passNo;
                u32 swOffset;
                u32 rejectSwRaster;
            } pcPersistCull;

            ctx->m_cmd->AddResourceBarrier(perView.m_persistCullBarrier);
            if (cullPass == CullingPass::First)
            {
                ctx->m_cmd->BufferClear(perView.m_allFilteredMeshletsAllCount.get(), 0);
                ctx->m_cmd->BufferClear(m_indirectDrawBuffer.get(), 0);
            }
            runUAVBufferBarrier(ctx->m_cmd, perView.m_allFilteredMeshletsAllCount.get());
            runUAVBufferBarrier(ctx->m_cmd, m_indirectDrawBuffer.get());
            // bind view buffer
            ctx->m_cmd->AttachBindlessRefCompute(m_persistentCullingPass, 1, perView.m_viewBindlessRef);
            ctx->m_cmd->AttachBindlessRefCompute(
                m_persistentCullingPass, 2, perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
            ctx->m_cmd->AttachBindlessRefCompute(m_persistentCullingPass, 3, m_persistCullDesc);
            ctx->m_cmd->AttachBindlessRefCompute(m_persistentCullingPass, 4, perView.m_allFilteredMeshletsDesc);
            ctx->m_cmd->AttachBindlessRefCompute(m_persistentCullingPass, 5, perView.m_instCullDesc);
            if (perView.m_viewType == PerFrameData::ViewType::Primary)
            {
                pcPersistCull.rejectSwRaster = 0;
            }
            else
            {
                pcPersistCull.rejectSwRaster = 1;
            }
            if (cullPass == CullingPass::First)
            {
                pcPersistCull.passNo   = 0;
                pcPersistCull.swOffset = perView.m_allFilteredMeshlets_SWOffset;

                ctx->m_cmd->SetPushConst(m_persistentCullingPass, 0, sizeof(PersistCullPushConst), &pcPersistCull);
                ctx->m_cmd->DispatchIndirect(perView.m_persistCullIndirectDispatch.get(), 0);
            }
            else if (cullPass == CullingPass::Second)
            {
                pcPersistCull.passNo   = 1;
                pcPersistCull.swOffset = perView.m_allFilteredMeshlets_SWOffset;

                ctx->m_cmd->SetPushConst(m_persistentCullingPass, 0, sizeof(PersistCullPushConst), &pcPersistCull);
                ctx->m_cmd->DispatchIndirect(perView.m_persistCullIndirectDispatch.get(), 6 * u32Size);
            }
        });

        auto& visPassHW =
            (filteredViewType == PerFrameData::ViewType::Primary) ? m_visibilityPassHW : m_depthOnlyVisibilityPassHW;
        visPassHW->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            // bind view buffer
            ctx->m_cmd->AttachBindlessRefGraphics(visPassHW, 1, perView.m_viewBindlessRef);
            ctx->m_cmd->AttachBindlessRefGraphics(visPassHW, 2, perframeData.m_allInstanceData.m_batchedObjBufRef);
            ctx->m_cmd->SetCullMode(RhiCullMode::Back);
            ctx->m_cmd->AttachBindlessRefGraphics(visPassHW, 3, perView.m_allFilteredMeshletsDesc);
            if (cullPass == CullingPass::First)
            {
                ctx->m_cmd->SetPushConst(visPassHW, 0, u32Size, &pcData[0]);
                ctx->m_cmd->DrawMeshTasksIndirect(perView.m_allFilteredMeshletsAllCount.get(), u32Size * 3, 1, 0);
            }
            else
            {
                ctx->m_cmd->SetPushConst(visPassHW, 0, u32Size, &pcData[1]);
                ctx->m_cmd->DrawMeshTasksIndirect(perView.m_allFilteredMeshletsAllCount.get(), u32Size * 0, 1, 0);
            }
        });

        auto& visPassSW = m_visibilityPassSW;

        visPassSW->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            runUAVBufferBarrier(cmd, perView.m_allFilteredMeshletsAllCount.get());

            // if this is the first pass, we need to clear the visibility buffer
            // seems cleaing visibility buffer is not necessary. Just clear the depth
            if (cullPass == CullingPass::First)
            {
                ctx->m_cmd->BufferClear(perView.m_visPassDepth_SW.get(),
                    0xffffffff); // clear to max_uint
                ctx->m_cmd->BufferClear(perView.m_visPassDepthCASLock_SW.get(), 0);
            }
            runUAVBufferBarrier(ctx->m_cmd, perView.m_visPassDepth_SW.get());
            runUAVBufferBarrier(ctx->m_cmd, perView.m_visPassDepthCASLock_SW.get());
            // bind view buffer
            ctx->m_cmd->AttachBindlessRefCompute(visPassSW, 1, perView.m_viewBindlessRef);
            ctx->m_cmd->AttachBindlessRefCompute(visPassSW, 2, perframeData.m_allInstanceData.m_batchedObjBufRef);
            ctx->m_cmd->AttachBindlessRefCompute(visPassSW, 3, perView.m_allFilteredMeshletsDesc);

            struct SWPushConst
            {
                u32 passNo;
                u32 depthBufferId;
                u32 visBufferId;
                u32 rtHeight;
                u32 rtWidth;
                u32 swOffset;
                u32 casBufferId;
            } pcsw;
            pcsw.passNo        = cullPass == CullingPass::First ? 0 : 1;
            pcsw.depthBufferId = perView.m_visPassDepth_SW->GetDescId();
            pcsw.visBufferId   = perView.m_visibilityBuffer_SW->GetDescId();
            pcsw.rtHeight      = perView.m_renderHeight;
            pcsw.rtWidth       = perView.m_renderWidth;
            pcsw.swOffset      = perView.m_allFilteredMeshlets_SWOffset;
            pcsw.casBufferId   = perView.m_visPassDepthCASLock_SW->GetDescId();
            ctx->m_cmd->SetPushConst(visPassSW, 0, sizeof(SWPushConst), &pcsw);
            if (cullPass == CullingPass::First)
            {
                ctx->m_cmd->DispatchIndirect(perView.m_allFilteredMeshletsAllCount.get(), u32Size * 13);
            }
            else
            {
                ctx->m_cmd->DispatchIndirect(perView.m_allFilteredMeshletsAllCount.get(), u32Size * 10);
            }
        });

        auto& combinePass = m_visibilityCombinePass;

        combinePass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            if (cullPass == CullingPass::First)
            {
                runImageBarrier(ctx->m_cmd, perView.m_visibilityBuffer_Combined.get(),
                    RhiResourceState::UnorderedAccess, { 0, 0, 1, 1 });
                runImageBarrier(ctx->m_cmd, perView.m_visibilityDepth_Combined.get(), RhiResourceState::UnorderedAccess,
                    { 0, 0, 1, 1 });
            }
            struct CombinePassPushConst
            {
                u32 hwVisUAVId;
                u32 hwDepthSRVId;
                u32 swVisUAVId;
                u32 swDepthUAVId;
                u32 rtWidth;
                u32 rtHeight;
                u32 outVisUAVId;
                u32 outDepthUAVId;
                u32 outMode;
            } pcCombine;
            pcCombine.rtWidth       = perView.m_renderWidth;
            pcCombine.rtHeight      = perView.m_renderHeight;
            pcCombine.outVisUAVId   = perView.m_visibilityBuffer_Combined->GetDescId();
            pcCombine.outDepthUAVId = perView.m_visibilityDepth_Combined->GetDescId();
            // Testing, not specifying sw ids
            pcCombine.hwVisUAVId   = perView.m_visibilityBuffer_HW->GetDescId();
            pcCombine.hwDepthSRVId = perView.m_visDepthIdSRV_HW->GetActiveId();
            pcCombine.swVisUAVId   = perView.m_visibilityBuffer_SW->GetDescId();
            pcCombine.swDepthUAVId = perView.m_visPassDepth_SW->GetDescId();
            pcCombine.outMode      = m_config->m_visualizationType == RendererVisualizationType::SwHwMaps;

            IF_CONSTEXPR auto wgSizeX = SyaroConfig::cCombineVisBufferThreadGroupSizeX;
            IF_CONSTEXPR auto wgSizeY = SyaroConfig::cCombineVisBufferThreadGroupSizeY;

            auto              tgX = Math::DivRoundUp(pcCombine.rtWidth, wgSizeX);
            auto              tgY = Math::DivRoundUp(pcCombine.rtHeight, wgSizeY);
            cmd->SetPushConst(combinePass, 0, sizeof(pcCombine), &pcCombine);
            ctx->m_cmd->Dispatch(tgX, tgY, 1);
        });

        if (cullPass == CullingPass::First)
        {
            cmd->BeginScope("Syaro: Cull Rasterize I");
        }
        else
        {
            cmd->BeginScope("Syaro: Cull Rasterize II");
        }
        cmd->BeginScope("Syaro: Instance Culling Pass");
        m_instanceCullingPass->Run(cmd, 0);
        cmd->EndScope();
        cmd->GlobalMemoryBarrier();
        cmd->BeginScope("Syaro: Persistent Culling Pass");
        m_persistentCullingPass->Run(cmd, 0);
        cmd->EndScope();
        cmd->GlobalMemoryBarrier();

        // SW Rasterize, TODO:Run in parallel with HW Rasterize. Not specify barrier
        // here
        if (perView.m_viewType == PerFrameData::ViewType::Primary)
        {
            cmd->BeginScope("Syaro: SW Rasterize");
            visPassSW->Run(cmd, 0);
            cmd->EndScope();
        }
        cmd->GlobalMemoryBarrier();
        // HW Rasterize
        cmd->BeginScope("Syaro: HW Rasterize");
        if (cullPass == CullingPass::First)
        {
            runUAVBufferBarrier(cmd, perView.m_allFilteredMeshletsAllCount.get());
            visPassHW->Run(cmd, perView.m_visRTs_HW.get(), 0);
        }
        else
        {
            runUAVBufferBarrier(cmd, perView.m_allFilteredMeshletsAllCount.get());
            visPassHW->Run(cmd, perView.m_visRTs2_HW.get(), 0);
        }
        cmd->EndScope();

        // Combine HW and SW results,
        if (perView.m_viewType == PerFrameData::ViewType::Primary)
        {
            cmd->GlobalMemoryBarrier();
            cmd->BeginScope("Syaro: SW Rasterize Merge");
            combinePass->Run(cmd, 0);
            cmd->EndScope();
        }

        // Run hi-z pass

        cmd->GlobalMemoryBarrier();
        cmd->BeginScope("Syaro: HiZ Pass");
        m_singlePassHiZProc->RunHiZPass(perView.m_spHiZData, cmd, perView.m_renderWidth, perView.m_renderHeight, false);
        cmd->EndScope();
        cmd->EndScope();
    }

    IFRIT_APIDECL void SyaroRenderer::RenderTriangleView(
        PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        auto  rhi         = m_app->GetRhi();
        auto& primaryView = GetPrimaryView(perframeData);
        SetupDebugPasses(perframeData, renderTargets);

        auto                      rtCfg = renderTargets->GetFormat();
        PipelineAttachmentConfigs cfg;
        cfg.m_colorFormats = rtCfg.m_colorFormats;
        cfg.m_depthFormat  = rtCfg.m_depthFormat;

        auto triangleView = m_triangleViewPass[cfg];
        struct PushConst
        {
            u32 visBufferSRV;
        } pc;
        pc.visBufferSRV = primaryView.m_visibilityBufferIdSRV_Combined->GetActiveId();
        RenderingUtil::EnqueueFullScreenPass(cmd, rhi, triangleView, renderTargets, {}, &pc, 1);
    }

    IFRIT_APIDECL void SyaroRenderer::RenderMaterialClassify(
        PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        auto rhi            = m_app->GetRhi();
        auto totalMaterials = SizeCast<u32>(perframeData.m_enabledEffects.size());

        u32  actualRtWidth = 0, actualRtHeight = 0;
        GetSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

        auto             width     = actualRtWidth;
        auto             height    = actualRtHeight;
        u32              pcData[3] = { width, height, totalMaterials };

        IF_CONSTEXPR u32 pTileWidth  = cMatClassQuadSize * cMatClassGroupSizeCountScatterX;
        IF_CONSTEXPR u32 pTileHeight = cMatClassQuadSize * cMatClassGroupSizeCountScatterY;

        // Counting
        auto             wgX = (width + pTileWidth - 1) / pTileWidth;
        auto             wgY = (height + pTileHeight - 1) / pTileHeight;
        m_matclassCountPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            ctx->m_cmd->BufferClear(perframeData.m_matClassCountBuffer.get(), 0);
            runUAVBufferBarrier(ctx->m_cmd, perframeData.m_matClassCountBuffer.get());

            ctx->m_cmd->AttachBindlessRefCompute(m_matclassCountPass, 1, perframeData.m_matClassDesc);
            ctx->m_cmd->SetPushConst(m_matclassCountPass, 0, u32Size * 3, &pcData[0]);
            ctx->m_cmd->Dispatch(wgX, wgY, 1);
        });

        // Reserving
        auto wgX2 = (totalMaterials + cMatClassGroupSizeReserveX - 1) / cMatClassGroupSizeReserveX;
        m_matclassReservePass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            ctx->m_cmd->AttachBindlessRefCompute(m_matclassReservePass, 1, perframeData.m_matClassDesc);
            ctx->m_cmd->SetPushConst(m_matclassReservePass, 0, u32Size * 3, &pcData[0]);
            ctx->m_cmd->Dispatch(wgX2, 1, 1);
        });

        // Scatter
        m_matclassScatterPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            ctx->m_cmd->AttachBindlessRefCompute(m_matclassScatterPass, 1, perframeData.m_matClassDesc);
            ctx->m_cmd->SetPushConst(m_matclassScatterPass, 0, u32Size * 3, &pcData[0]);
            ctx->m_cmd->Dispatch(wgX, wgY, 1);
        });

        // Start rendering
        cmd->BeginScope("Syaro: Material Classification");
        cmd->GlobalMemoryBarrier();
        m_matclassCountPass->Run(cmd, 0);
        cmd->AddResourceBarrier(perframeData.m_matClassBarrier);
        cmd->GlobalMemoryBarrier();
        m_matclassReservePass->Run(cmd, 0);
        cmd->AddResourceBarrier(perframeData.m_matClassBarrier);
        cmd->GlobalMemoryBarrier();
        m_matclassScatterPass->Run(cmd, 0);
        cmd->AddResourceBarrier(perframeData.m_matClassBarrier);
        cmd->GlobalMemoryBarrier();
        cmd->EndScope();
    }

    IFRIT_APIDECL void SyaroRenderer::SetupDefaultEmitGBufferPass()
    {
        auto rhi = m_app->GetRhi();
        m_defaultEmitGBufferPass =
            RenderingUtil::CreateComputePass(rhi, "Syaro/Syaro.EmitGBuffer.Default.comp.glsl", 6, 3);
    }

    IFRIT_APIDECL void SyaroRenderer::SphizBufferSetup(PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        for (u32 k = 0; k < perframeData.m_views.size(); k++)
        {
            auto& perView = perframeData.m_views[k];
            auto  width   = perView.m_renderWidth;
            auto  height  = perView.m_renderHeight;
            auto  rWidth  = 1 << (int)std::ceil(std::log2(width));
            auto  rHeight = 1 << (int)std::ceil(std::log2(height));
            // Max-HiZ required by instance cull
            if (m_singlePassHiZProc->CheckResourceToRebuild(perView.m_spHiZData, width, height))
            {
                m_singlePassHiZProc->PrepareHiZResources(perView.m_spHiZData, perView.m_visibilityDepth_Combined.get(),
                    m_immRes.m_linearSampler.get(), rWidth, rHeight);
            }
            // Min-HiZ required by ssgi
            if (m_singlePassHiZProc->CheckResourceToRebuild(perView.m_spHiZDataMin, width, height))
            {
                m_singlePassHiZProc->PrepareHiZResources(perView.m_spHiZDataMin,
                    perView.m_visibilityDepth_Combined.get(), m_immRes.m_linearSampler.get(), rWidth, rHeight);
            }
        }
    }

    IFRIT_APIDECL void SyaroRenderer::VisibilityBufferSetup(PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        auto rhi = m_app->GetRhi();

        for (u32 k = 0; k < perframeData.m_views.size(); k++)
        {
            auto& perView    = perframeData.m_views[k];
            bool  createCond = (perView.m_visibilityBuffer_Combined == nullptr);

            u32   actualRtw = 0, actualRth = 0;
            GetSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);

            auto visHeight = perView.m_renderHeight;
            auto visWidth  = perView.m_renderWidth;
            if (visHeight == 0 || visWidth == 0)
            {
                if (perView.m_viewType == PerFrameData::ViewType::Primary)
                {
                    // use render target size
                    visHeight = actualRth;
                    visWidth  = actualRtw;
                }
                else if (perView.m_viewType == PerFrameData::ViewType::Shadow)
                {
                    iError("Shadow view has no size");
                    std::abort();
                }
            }

            if (!createCond)
            {
                return;
            }
            // It seems nanite's paper uses R32G32 for mesh visibility, but
            // I wonder the depth is implicitly calculated from the depth buffer
            // so here I use R32 for visibility buffer
            // Update: Now it's required for SW rasterizer. But I use a separate
            // buffer

            // For HW rasterizer
            if IF_CONSTEXPR (true)
            {
                auto visBufferHW              = rhi->CreateTexture2D("Syaro_VisBufferHW", visWidth, visHeight,
                                 PerFrameData::c_visibilityFormat, kbImUsage_UAV_SRV_RT, true);
                auto visDepthHW               = rhi->CreateDepthTexture("Syaro_VisDepthHW", visWidth, visHeight, false);
                auto visDepthSampler          = rhi->CreateTrivialSampler();
                perView.m_visibilityBuffer_HW = visBufferHW;

                // first pass rts
                perView.m_visPassDepth_HW = visDepthHW;
                perView.m_visDepthIdSRV_HW =
                    rhi->RegisterCombinedImageSampler(perView.m_visPassDepth_HW.get(), m_immRes.m_linearSampler.get());
                perView.m_visDepthRT_HW =
                    rhi->CreateRenderTargetDepthStencil(visDepthHW.get(), { {}, 1.0f }, RhiRenderTargetLoadOp::Clear);

                perView.m_visRTs_HW = rhi->CreateRenderTargets();
                if (perView.m_viewType == PerFrameData::ViewType::Primary)
                {
                    perView.m_visColorRT_HW = rhi->CreateRenderTarget(
                        visBufferHW.get(), { { 0, 0, 0, 0 } }, RhiRenderTargetLoadOp::Clear, 0, 0);
                    perView.m_visRTs_HW->SetColorAttachments({ perView.m_visColorRT_HW.get() });
                }
                else
                {
                    // shadow passes do not need color attachment
                    perView.m_visRTs_HW->SetColorAttachments({});
                }
                perView.m_visRTs_HW->SetDepthStencilAttachment(perView.m_visDepthRT_HW.get());
                perView.m_visRTs_HW->SetRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));
                if (perView.m_viewType == PerFrameData::ViewType::Shadow)
                {
                    auto rtHeight = (perView.m_renderHeight);
                    auto rtWidth  = (perView.m_renderWidth);
                    perView.m_visRTs_HW->SetRenderArea({ 0, 0, rtWidth, rtHeight });
                }

                // second pass rts
                perView.m_visDepthRT2_HW =
                    rhi->CreateRenderTargetDepthStencil(visDepthHW.get(), { {}, 1.0f }, RhiRenderTargetLoadOp::Load);
                perView.m_visRTs2_HW = rhi->CreateRenderTargets();
                if (perView.m_viewType == PerFrameData::ViewType::Primary)
                {
                    perView.m_visColorRT2_HW = rhi->CreateRenderTarget(
                        visBufferHW.get(), { { 0, 0, 0, 0 } }, RhiRenderTargetLoadOp::Load, 0, 0);
                    perView.m_visRTs2_HW->SetColorAttachments({ perView.m_visColorRT2_HW.get() });
                }
                else
                {
                    // shadow passes do not need color attachment
                    perView.m_visRTs2_HW->SetColorAttachments({});
                }
                perView.m_visRTs2_HW->SetDepthStencilAttachment(perView.m_visDepthRT2_HW.get());
                perView.m_visRTs2_HW->SetRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));
                if (perView.m_viewType == PerFrameData::ViewType::Shadow)
                {
                    auto rtHeight = (perView.m_renderHeight);
                    auto rtWidth  = (perView.m_renderWidth);
                    perView.m_visRTs2_HW->SetRenderArea({ 0, 0, rtWidth, rtHeight });
                }
            }

            // For SW rasterizer
            if (SYARO_ENABLE_SW_RASTERIZER && perView.m_viewType == PerFrameData::ViewType::Primary)
            {
                perView.m_visibilityBuffer_SW = rhi->CreateTexture2D("Syaro_VisBufferSW", visWidth, visHeight,
                    PerFrameData::c_visibilityFormat, kbImUsage_UAV_SRV, true);
                perView.m_visPassDepth_SW     = rhi->CreateBufferDevice(
                    "Syaro_VisDepthSW", u64Size * visWidth * visHeight, kbBufUsage_SSBO_CopyDest, true);
                perView.m_visPassDepthCASLock_SW = rhi->CreateBufferDevice(
                    "Syaro_VisCasSW", f32Size * visWidth * visHeight, kbBufUsage_SSBO_CopyDest, true);
            }

            // For combined buffer
            if (SYARO_ENABLE_SW_RASTERIZER && perView.m_viewType == PerFrameData::ViewType::Primary)
            {
                perView.m_visibilityBuffer_Combined = rhi->CreateTexture2D("Syaro_VisBufferComb", visWidth, visHeight,
                    PerFrameData::c_visibilityFormat, kbImUsage_UAV_SRV, true);
                perView.m_visibilityDepth_Combined  = rhi->CreateTexture2D(
                    "Syaro_VisDepthComb", visWidth, visHeight, kbImFmt_R32F, kbImUsage_UAV_SRV, true);
                perView.m_visibilityBufferIdSRV_Combined = rhi->RegisterCombinedImageSampler(
                    perView.m_visibilityBuffer_Combined.get(), m_immRes.m_nearestSampler.get());
                perView.m_visibilityDepthIdSRV_Combined = rhi->RegisterCombinedImageSampler(
                    perView.m_visibilityDepth_Combined.get(), m_immRes.m_linearSampler.get());
            }
            else
            {
                perView.m_visibilityDepth_Combined       = perView.m_visPassDepth_HW;
                perView.m_visibilityBuffer_Combined      = perView.m_visibilityBuffer_HW;
                perView.m_visibilityDepthIdSRV_Combined  = perView.m_visDepthIdSRV_HW;
                perView.m_visibilityBufferIdSRV_Combined = rhi->RegisterCombinedImageSampler(
                    perView.m_visibilityBuffer_Combined.get(), m_immRes.m_linearSampler.get());
            }
        }
    }

    IFRIT_APIDECL void SyaroRenderer::RenderDefaultEmitGBuffer(
        PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        auto numMaterials = perframeData.m_enabledEffects.size();
        m_defaultEmitGBufferPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            // first transition all gbuffer textures to UAV/Common
            runImageBarrier(ctx->m_cmd, perframeData.m_gbuffer.m_albedo_materialFlags.get(), RhiResourceState::Common,
                { 0, 0, 1, 1 });
            runImageBarrier(
                ctx->m_cmd, perframeData.m_gbuffer.m_normal_smoothness.get(), RhiResourceState::Common, { 0, 0, 1, 1 });
            runImageBarrier(
                ctx->m_cmd, perframeData.m_gbuffer.m_emissive.get(), RhiResourceState::Common, { 0, 0, 1, 1 });
            runImageBarrier(ctx->m_cmd, perframeData.m_gbuffer.m_specular_occlusion.get(), RhiResourceState::Common,
                { 0, 0, 1, 1 });
            runImageBarrier(
                ctx->m_cmd, perframeData.m_gbuffer.m_shadowMask.get(), RhiResourceState::Common, { 0, 0, 1, 1 });

            // Clear gbuffer textures
            ctx->m_cmd->ClearUAVTexFloat(
                perframeData.m_gbuffer.m_albedo_materialFlags.get(), { 0, 0, 1, 1 }, { 0.0f, 0.0f, 0.0f, 0.0f });
            ctx->m_cmd->ClearUAVTexFloat(
                perframeData.m_gbuffer.m_normal_smoothness.get(), { 0, 0, 1, 1 }, { 0.0f, 0.0f, 0.0f, 0.0f });
            ctx->m_cmd->ClearUAVTexFloat(
                perframeData.m_gbuffer.m_emissive.get(), { 0, 0, 1, 1 }, { 0.0f, 0.0f, 0.0f, 0.0f });
            ctx->m_cmd->ClearUAVTexFloat(
                perframeData.m_gbuffer.m_specular_occlusion.get(), { 0, 0, 1, 1 }, { 0.0f, 0.0f, 0.0f, 0.0f });
            ctx->m_cmd->ClearUAVTexFloat(
                perframeData.m_gbuffer.m_shadowMask.get(), { 0, 0, 1, 1 }, { 0.0f, 0.0f, 0.0f, 0.0f });
            ctx->m_cmd->AddResourceBarrier(perframeData.m_gbuffer.m_gbufferBarrier);

            // For each material, make
            // one dispatch
            u32 actualRtw = 0, actualRth = 0;
            GetSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);
            u32   pcData[3]   = { 0, actualRtw, actualRth };
            auto& primaryView = GetPrimaryView(perframeData);
            for (int i = 0; i < numMaterials; i++)
            {
                ctx->m_cmd->AttachBindlessRefCompute(m_defaultEmitGBufferPass, 1, perframeData.m_matClassDesc);
                ctx->m_cmd->AttachBindlessRefCompute(m_defaultEmitGBufferPass, 2, perframeData.m_gbuffer.m_gbufferDesc);
                ctx->m_cmd->AttachBindlessRefCompute(m_defaultEmitGBufferPass, 3, primaryView.m_viewBindlessRef);
                ctx->m_cmd->AttachBindlessRefCompute(
                    m_defaultEmitGBufferPass, 4, perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
                ctx->m_cmd->AttachBindlessRefCompute(
                    m_defaultEmitGBufferPass, 5, primaryView.m_allFilteredMeshletsDesc);
                ctx->m_cmd->AttachBindlessRefCompute(m_defaultEmitGBufferPass, 6, perframeData.m_velocityMaterialDesc);

                pcData[0] = i;
                ctx->m_cmd->SetPushConst(m_defaultEmitGBufferPass, 0, u32Size * 3, &pcData[0]);
                RhiResourceBarrier barrierCountBuffer;
                barrierCountBuffer.m_type         = RhiBarrierType::UAVAccess;
                barrierCountBuffer.m_uav.m_buffer = perframeData.m_matClassCountBuffer.get();
                barrierCountBuffer.m_uav.m_type   = RhiResourceType::Buffer;

                ctx->m_cmd->AddResourceBarrier({ barrierCountBuffer });
                ctx->m_cmd->DispatchIndirect(perframeData.m_matClassIndirectDispatchBuffer.get(), i * u32Size * 4);
                ctx->m_cmd->AddResourceBarrier(perframeData.m_gbuffer.m_gbufferBarrier);
            }
        });
        auto rhi = m_app->GetRhi();
        cmd->BeginScope("Syaro: Emit  GBuffer");
        m_defaultEmitGBufferPass->Run(cmd, 0);
        cmd->EndScope();
    }

    IFRIT_APIDECL void SyaroRenderer::RenderAmbientOccl(
        PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        auto primaryView = GetPrimaryView(perframeData);

        auto albedoSamp        = perframeData.m_gbuffer.m_albedo_materialFlags_sampId;
        auto normalSamp        = perframeData.m_gbuffer.m_normal_smoothness_sampId;
        auto depthSamp         = primaryView.m_visibilityDepthIdSRV_Combined;
        auto perframe          = primaryView.m_viewBufferId;
        auto ao                = perframeData.m_gbuffer.m_specular_occlusion->GetDescId();
        auto aoIntermediate    = perframeData.m_gbuffer.m_specular_occlusion_intermediate->GetDescId();
        auto aoIntermediateSRV = perframeData.m_gbuffer.m_specular_occlusion_intermediate_sampId;

        auto aoRT     = perframeData.m_gbuffer.m_specular_occlusion_RTs;
        auto fsr2samp = perframeData.m_fsr2Data.m_fsr2OutputSRVId;

        u32  width  = primaryView.m_renderWidth;
        u32  height = primaryView.m_renderHeight;

        auto aoBlurFunc = [&]() {
            // Blurring AO
            auto rhi = m_app->GetRhi();
            cmd->GlobalMemoryBarrier();
            auto colorRT1 = rhi->CreateRenderTarget(perframeData.m_gbuffer.m_specular_occlusion_intermediate.get(),
                { { 0, 0, 0, 0 } }, RhiRenderTargetLoadOp::Clear, 0, 0);
            auto colorRT2 = rhi->CreateRenderTarget(perframeData.m_gbuffer.m_specular_occlusion.get(),
                { { 0, 0, 0, 0 } }, RhiRenderTargetLoadOp::Clear, 0, 0);
            auto rt1      = rhi->CreateRenderTargets();
            rt1->SetColorAttachments({ colorRT1.get() });
            rt1->SetRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));
            auto rt2 = rhi->CreateRenderTargets();
            rt2->SetColorAttachments({ colorRT2.get() });
            rt2->SetRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));

            m_gaussianHori->RenderPostFx(cmd, rt1.get(), perframeData.m_gbuffer.m_specular_occlusion_sampId.get(), 5);
            m_gaussianVert->RenderPostFx(
                cmd, rt2.get(), perframeData.m_gbuffer.m_specular_occlusion_intermediate_sampId.get(), 5);
        };

        cmd->BeginScope("Syaro: Ambient Occlusion");
        cmd->AddResourceBarrier(
            { perframeData.m_gbuffer.m_normal_smoothnessBarrier, perframeData.m_gbuffer.m_specular_occlusionBarrier });
        if (m_config->m_indirectLightingType == IndirectLightingType::HBAO)
        {
            m_aoPass->RenderHBAO(cmd, width, height, depthSamp.get(), normalSamp.get(), ao, perframe.get());
            cmd->AddResourceBarrier({ perframeData.m_gbuffer.m_specular_occlusionBarrier });
            aoBlurFunc();
        }
        else if (m_config->m_indirectLightingType == IndirectLightingType::SSGI)
        {
            m_aoPass->RenderHBAO(cmd, width, height, depthSamp.get(), normalSamp.get(), ao, perframe.get());
            cmd->AddResourceBarrier({ perframeData.m_gbuffer.m_specular_occlusionBarrier });
            aoBlurFunc();
            cmd->GlobalMemoryBarrier();
            m_singlePassHiZProc->RunHiZPass(primaryView.m_spHiZDataMin, cmd, width, height, true);
            cmd->GlobalMemoryBarrier();
            m_aoPass->RenderSSGI(cmd, width, height, primaryView.m_viewBufferId.get(),
                primaryView.m_spHiZDataMin.m_hizRefBuffer->GetDescId(),
                primaryView.m_spHiZData.m_hizRefBuffer->GetDescId(), normalSamp.get(), aoIntermediate, albedoSamp.get(),
                primaryView.m_spHiZDataMin.m_hizWidth, primaryView.m_spHiZDataMin.m_hizHeight,
                primaryView.m_spHiZDataMin.m_hizIters, m_immRes.m_blueNoiseSRV.get());

            cmd->GlobalMemoryBarrier();
            m_jointBilateralFilter->RenderPostFx(
                cmd, aoRT.get(), aoIntermediateSRV.get(), normalSamp.get(), depthSamp.get(), 0);
        }
        cmd->GlobalMemoryBarrier();
        cmd->EndScope();
    }

    IFRIT_APIDECL void SyaroRenderer::GatherAllInstances(PerFrameData& perframeData)
    {
        u32 totalInstances = 0;
        u32 totalMeshlets  = 0;
        for (auto x : perframeData.m_enabledEffects)
        {
            auto& effect = perframeData.m_shaderEffectData[x];
            totalInstances += SizeCast<u32>(effect.m_objectData.size());
        }
        auto rhi = m_app->GetRhi();
        if (perframeData.m_allInstanceData.m_lastObjectCount != totalInstances)
        {
            perframeData.m_allInstanceData.m_lastObjectCount = totalInstances;
            perframeData.m_allInstanceData.m_batchedObjectData =
                rhi->CreateBufferCoherent(totalInstances * sizeof(PerObjectData), kbBufUsage_SSBO);
            perframeData.m_allInstanceData.m_batchedObjBufRef = rhi->createBindlessDescriptorRef();
            auto buf                                          = perframeData.m_allInstanceData.m_batchedObjectData;
            perframeData.m_allInstanceData.m_batchedObjBufRef->AddStorageBuffer(buf.get(), 0);
        }
        perframeData.m_allInstanceData.m_objectData.resize(totalInstances);

        for (auto i = 0; auto& x : perframeData.m_enabledEffects)
        {
            auto& effect = perframeData.m_shaderEffectData[x];
            for (auto& obj : effect.m_objectData)
            {
                perframeData.m_allInstanceData.m_objectData[i] = obj;
                i++;
            }
            u32 objDataSize = static_cast<u32>(effect.m_objectData.size());
            u32 matSize     = static_cast<u32>(perframeData.m_shaderEffectData[x].m_materials.size());
            for (u32 k = 0; k < matSize; k++)
            {
                auto mesh        = perframeData.m_shaderEffectData[x].m_meshes[k]->LoadMeshUnsafe();
                auto lv0Meshlets = mesh->m_numMeshletsEachLod[0];
                auto lv1Meshlets = 0;
                if (mesh->m_numMeshletsEachLod.size() > 1)
                {
                    lv1Meshlets = mesh->m_numMeshletsEachLod[1];
                }
                totalMeshlets += (lv0Meshlets + lv1Meshlets);
            }
        }
        auto activeBuf = perframeData.m_allInstanceData.m_batchedObjectData->GetActiveBuffer();
        activeBuf->MapMemory();
        activeBuf->WriteBuffer(perframeData.m_allInstanceData.m_objectData.data(),
            SizeCast<u32>(perframeData.m_allInstanceData.m_objectData.size() * sizeof(PerObjectData)), 0);
        activeBuf->FlushBuffer();
        activeBuf->UnmapMemory();

        RhiBufferRef allFilteredMeshletsHW = nullptr;
        RhiBufferRef allFilteredMeshletsSW = nullptr;
        for (u32 k = 0; k < perframeData.m_views.size(); k++)
        {
            auto& perView = perframeData.m_views[k];
            if (perView.m_allFilteredMeshletsAllCount == nullptr)
            {
                perView.m_allFilteredMeshletsAllCount =
                    rhi->CreateBufferDevice("Syaro_FilteredClusterCnt", u32Size * 20, kbBufUsage_Indirect, true);
            }
            if (perView.m_allFilteredMeshletsMaxCount < totalMeshlets)
            {
                perView.m_allFilteredMeshletsMaxCount = totalMeshlets;
                IF_CONSTEXPR u32 bufMultiplier        = 1 + SYARO_ENABLE_SW_RASTERIZER;
                if (allFilteredMeshletsHW == nullptr)
                {
                    allFilteredMeshletsHW = rhi->CreateBufferDevice("Syaro_FilteredClusterHW",
                        totalMeshlets * bufMultiplier * sizeof(Vector2i) + 2, kbBufUsage_SSBO_CopyDest, true);
                }
                perView.m_allFilteredMeshletsHW = allFilteredMeshletsHW;

#if SYARO_ENABLE_SW_RASTERIZER
                if (allFilteredMeshletsSW == nullptr)
                {
                    allFilteredMeshletsSW = allFilteredMeshletsHW;
                }
                perView.m_allFilteredMeshlets_SWOffset = totalMeshlets + 1;
                perView.m_allFilteredMeshletsSW        = allFilteredMeshletsSW;
#endif

                perView.m_allFilteredMeshletsDesc = rhi->createBindlessDescriptorRef();
                perView.m_allFilteredMeshletsDesc->AddStorageBuffer(perView.m_allFilteredMeshletsHW.get(), 0);
#if SYARO_ENABLE_SW_RASTERIZER
                perView.m_allFilteredMeshletsDesc->AddStorageBuffer(perView.m_allFilteredMeshletsSW.get(), 1);
#else
                perView.m_allFilteredMeshletsDesc->AddStorageBuffer(perView.m_allFilteredMeshletsHW, 1);
#endif
                perView.m_allFilteredMeshletsDesc->AddStorageBuffer(perView.m_allFilteredMeshletsAllCount.get(), 2);
            }
        }
    }

    IFRIT_APIDECL void SyaroRenderer::PrepareAggregatedShadowData(PerFrameData& perframeData)
    {
        auto& shadowData = perframeData.m_shadowData2;
        auto  rhi        = m_app->GetRhi();

        if (shadowData.m_allShadowData == nullptr)
        {
            shadowData.m_allShadowData = rhi->CreateBufferCoherent(
                256 * sizeof(decltype(shadowData.m_shadowViews)::value_type), kbBufUsage_SSBO_CopyDest);
            shadowData.m_allShadowDataId = rhi->RegisterStorageBufferShared(shadowData.m_allShadowData.get());
        }
        for (auto d = 0; auto& x : shadowData.m_shadowViews)
        {
            for (auto i = 0u; i < x.m_csmSplits; i++)
            {
                auto idx       = x.m_viewMapping[i];
                x.m_texRef[i]  = perframeData.m_views[idx].m_visibilityDepthIdSRV_Combined->GetActiveId();
                x.m_viewRef[i] = perframeData.m_views[idx].m_viewBufferId->GetActiveId();
            }
        }
        auto p = shadowData.m_allShadowData->GetActiveBuffer();
        p->MapMemory();
        p->WriteBuffer(shadowData.m_shadowViews.data(),
            SizeCast<u32>(shadowData.m_shadowViews.size() * sizeof(decltype(shadowData.m_shadowViews)::value_type)), 0);
        p->FlushBuffer();
        p->UnmapMemory();

        auto mainView     = GetPrimaryView(perframeData);
        auto mainRtWidth  = mainView.m_renderWidth;
        auto mainRtHeight = mainView.m_renderHeight;
        if (perframeData.m_deferShadowMask == nullptr)
        {
            perframeData.m_deferShadowMask = rhi->CreateTexture2D(
                "Syaro_DeferShadow", mainRtWidth, mainRtHeight, kbImFmt_RGBA32F, kbImUsage_UAV_SRV_RT, true);

            perframeData.m_deferShadowMaskRT  = rhi->CreateRenderTarget(perframeData.m_deferShadowMask.get(),
                 { { 0.0f, 0.0f, 0.0f, 1.0f } }, RhiRenderTargetLoadOp::Clear, 0, 0);
            perframeData.m_deferShadowMaskRTs = rhi->CreateRenderTargets();
            perframeData.m_deferShadowMaskRTs->SetColorAttachments({ perframeData.m_deferShadowMaskRT.get() });
            perframeData.m_deferShadowMaskRTs->SetRenderArea({ 0, 0, u32(mainRtWidth), u32(mainRtHeight) });
            perframeData.m_deferShadowMaskId =
                rhi->RegisterCombinedImageSampler(perframeData.m_deferShadowMask.get(), m_immRes.m_linearSampler.get());
        }
    }

    IFRIT_APIDECL void SyaroRenderer::Fsr2Setup(PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        auto rhi       = m_app->GetRhi();
        u32  actualRtw = 0, actualRth = 0;
        u32  outputRtw = 0, outputRth = 0;
        GetSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);

        auto outputArea = renderTargets->GetRenderArea();
        outputRtw       = outputArea.width;
        outputRth       = outputArea.height;
        if (perframeData.m_fsr2Data.m_fsr2Output != nullptr)
            return;
        perframeData.m_fsr2Data.m_fsr2Output =
            rhi->CreateTexture2D("Syaro_FSR2Out", outputRtw, outputRth, kbImFmt_RGBA16F, kbImUsage_UAV_SRV, true);

        perframeData.m_fsr2Data.m_fsr2OutputSRVId = rhi->RegisterCombinedImageSampler(
            perframeData.m_fsr2Data.m_fsr2Output.get(), m_immRes.m_linearSampler.get());

        if (m_config->m_antiAliasingType == AntiAliasingType::FSR2)
        {
            Graphics::Rhi::FSR2::RhiFSR2InitialzeArgs args;
            args.displayHeight   = outputRth;
            args.displayWidth    = outputRtw;
            args.maxRenderWidth  = actualRtw;
            args.maxRenderHeight = actualRth;
            m_fsr2proc->Init(args);
        }
    }

    IFRIT_APIDECL void SyaroRenderer::TaaHistorySetup(PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        auto rhi = m_app->GetRhi();

        u32  actualRtw = 0, actualRth = 0;
        GetSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);

        auto width  = actualRtw;
        auto height = actualRth;

        auto needRecreate = (perframeData.m_taaHistory.size() == 0);
        if (!needRecreate)
        {
            needRecreate =
                (perframeData.m_taaHistory[0].m_width != width || perframeData.m_taaHistory[0].m_height != height);
        }
        if (!needRecreate)
        {
            return;
        }
        perframeData.m_taaHistory.clear();
        perframeData.m_taaHistory.resize(2);
        perframeData.m_taaHistory[0].m_width  = width;
        perframeData.m_taaHistory[0].m_height = height;
        perframeData.m_taaHistoryDesc         = rhi->createBindlessDescriptorRef();
        auto rtFormat                         = renderTargets->GetFormat();

        perframeData.m_taaUnresolved =
            rhi->CreateTexture2D("Syaro_TAAUnresolved", width, height, cTAAFormat, kbImUsage_UAV_SRV_RT_CopySrc, true);

        perframeData.m_taaHistoryDesc->AddCombinedImageSampler(
            perframeData.m_taaUnresolved.get(), m_immRes.m_linearSampler.get(), 0);
        for (int i = 0; i < 2; i++)
        {
            // TODO: choose formats
            perframeData.m_taaHistory[i].m_colorRT =
                rhi->CreateTexture2D("Syaro_TAAHistory", width, height, cTAAFormat, kbImUsage_UAV_SRV_RT_CopySrc, true);
            // perframeData.m_taaHistory[i].m_colorRTId = rhi->RegisterUAVImage(perframeData.m_taaUnresolved.get(), {0,
            // 0, 1, 1});
            perframeData.m_taaHistory[i].m_colorRTIdSRV =
                rhi->RegisterCombinedImageSampler(perframeData.m_taaUnresolved.get(), m_immRes.m_linearSampler.get());

            // TODO: clear values
            perframeData.m_taaHistory[i].m_colorRTRef = rhi->CreateRenderTarget(
                perframeData.m_taaUnresolved.get(), { { 0, 0, 0, 0 } }, RhiRenderTargetLoadOp::Clear, 0, 0);

            RhiAttachmentBlendInfo blendInfo;
            blendInfo.m_blendEnable         = false;
            blendInfo.m_srcColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
            blendInfo.m_dstColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blendInfo.m_colorBlendOp        = RhiBlendOp::RHI_BLEND_OP_ADD;
            blendInfo.m_alphaBlendOp        = RhiBlendOp::RHI_BLEND_OP_ADD;
            blendInfo.m_srcAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
            blendInfo.m_dstAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            perframeData.m_taaHistory[i].m_colorRTRef->SetBlendInfo(blendInfo);

            perframeData.m_taaHistory[i].m_rts = rhi->CreateRenderTargets();
            perframeData.m_taaHistory[i].m_rts->SetColorAttachments(
                { perframeData.m_taaHistory[i].m_colorRTRef.get() });
            perframeData.m_taaHistory[i].m_rts->SetRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));

            perframeData.m_taaHistoryDesc->AddCombinedImageSampler(
                perframeData.m_taaHistory[i].m_colorRT.get(), m_immRes.m_linearSampler.get(), i + 1);
        }
    }

    IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission> SyaroRenderer::Render(PerFrameData& perframeData,
        SyaroRenderer::RenderTargets* renderTargets, const std::vector<SyaroRenderer::GPUCommandSubmission*>& cmdToWait)
    {

        // According to
        // lunarg(https://www.lunasdk.org/manual/rhi/command_queues_and_command_buffers/)
        // graphics queue can accept dispatch and transfer commands,
        // but compute queue can only accept compute/transfer commands.
        // Following posts suggests to reduce command buffer submission to
        // improve performance
        // https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/
        // https://gpuopen.com/learn/rdna-performance-guide/#command-buffers

        auto start = std::chrono::high_resolution_clock::now();

        VisibilityBufferSetup(perframeData, renderTargets);
        auto& primaryView = GetPrimaryView(perframeData);
        BuildPipelines(perframeData, GraphicsShaderPassType::Opaque, primaryView.m_visRTs_HW.get());
        PrepareDeviceResources(perframeData, renderTargets);
        GatherAllInstances(perframeData);

        RecreateInstanceCullingBuffers(perframeData, SizeCast<u32>(perframeData.m_allInstanceData.m_objectData.size()));
        DepthTargetsSetup(perframeData, renderTargets);
        MaterialClassifyBufferSetup(perframeData, renderTargets);
        RecreateGBuffers(perframeData, renderTargets);
        SphizBufferSetup(perframeData, renderTargets);
        TaaHistorySetup(perframeData, renderTargets);
        Fsr2Setup(perframeData, renderTargets);

        auto start1 = std::chrono::high_resolution_clock::now();
        PrepareAggregatedShadowData(perframeData);
        auto                            end1     = std::chrono::high_resolution_clock::now();
        auto                            elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

        // Then draw
        auto                            rhi = m_app->GetRhi();
        auto                            dq  = rhi->GetQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);

        std::vector<RhiTaskSubmission*> cmdToWaitBkp = cmdToWait;
        std::unique_ptr<RhiTaskSubmission> pbrAtmoTask;
        if (perframeData.m_atmosphereData == nullptr)
        {
            // Need to create an atmosphere output texture
            u32 actualRtw = 0, actualRth = 0;
            GetSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);
            perframeData.m_atmoOutput = rhi->CreateTexture2D(
                "Syaro_AtmoOutput", actualRtw, actualRth, kbImFmt_RGBA32F, kbImUsage_UAV_SRV, true);

            // Precompute only once
            pbrAtmoTask  = this->m_atmosphereRenderer->RenderInternal(perframeData, cmdToWait);
            cmdToWaitBkp = { pbrAtmoTask.get() };
        }

        auto end0     = std::chrono::high_resolution_clock::now();
        auto elapsed0 = std::chrono::duration_cast<std::chrono::microseconds>(end0 - start);

        start         = std::chrono::high_resolution_clock::now();
        auto mainTask = dq->RunAsyncCommand(
            [&](const RhiCommandList* cmd) {
                for (u32 i = 0; i < perframeData.m_views.size(); i++)
                {
                    if (perframeData.m_views[i].m_viewType == PerFrameData::ViewType::Shadow
                        && m_config->m_visualizationType == RendererVisualizationType::Default)
                    {
                        if ((m_renderRole & SyaroRenderRole::Shadowing))
                        {
                            cmd->BeginScope("Syaro: Draw Call, Shadow View");
                            cmd->GlobalMemoryBarrier();
                            auto& perView = perframeData.m_views[i];
                            RenderTwoPassOcclCulling(CullingPass::First, perframeData, renderTargets, cmd,
                                PerFrameData::ViewType::Shadow, i);
                            RenderTwoPassOcclCulling(CullingPass::Second, perframeData, renderTargets, cmd,
                                PerFrameData::ViewType::Shadow, i);
                            cmd->EndScope();
                        }
                    }
                    else if (perframeData.m_views[i].m_viewType == PerFrameData::ViewType::Primary)
                    {
                        if ((m_renderRole & SyaroRenderRole::GBuffer))
                        {
                            cmd->BeginScope("Syaro: Draw Call, Main View");
                            RenderTwoPassOcclCulling(CullingPass::First, perframeData, renderTargets, cmd,
                                PerFrameData::ViewType::Primary, ~0u);
                            RenderTwoPassOcclCulling(CullingPass::Second, perframeData, renderTargets, cmd,
                                PerFrameData::ViewType::Primary, ~0u);
                            cmd->GlobalMemoryBarrier();
                            if (m_config->m_visualizationType != RendererVisualizationType::Default)
                            {
                                return;
                            }
                            RenderEmitDepthTargets(perframeData, renderTargets, cmd);
                            cmd->GlobalMemoryBarrier();
                            RenderMaterialClassify(perframeData, renderTargets, cmd);
                            cmd->GlobalMemoryBarrier();
                            RenderDefaultEmitGBuffer(perframeData, renderTargets, cmd);
                            cmd->GlobalMemoryBarrier();
                            if (m_renderRole == SyaroRenderRole::FullProcess)
                            {
                                RenderAmbientOccl(perframeData, renderTargets, cmd);
                            }
                            cmd->EndScope();
                        }
                    }
                }
            },
            cmdToWaitBkp, {});

        if (m_renderRole & SyaroRenderRole::Shading)
        {
            if (m_config->m_visualizationType != RendererVisualizationType::Default)
            {
                auto deferredTask = dq->RunAsyncCommand(
                    [&](const RhiCommandList* cmd) {
                        if (m_config->m_visualizationType == RendererVisualizationType::Triangle
                            || m_config->m_visualizationType == RendererVisualizationType::SwHwMaps)
                        {
                            cmd->GlobalMemoryBarrier();
                            RenderTriangleView(perframeData, renderTargets, cmd);
                            return;
                        }
                    },
                    { mainTask.get() }, {});
                return deferredTask;
            }
            else
            {
                auto deferredTask = dq->RunAsyncCommand(
                    [&](const RhiCommandList* cmd) { SetupAndRunFrameGraph(perframeData, renderTargets, cmd); },
                    { mainTask.get() }, {});
                return deferredTask;
            }
        }
        else
        {
            return mainTask;
        }
    }

    IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission> SyaroRenderer::Render(Scene* scene,
        Camera* camera, RenderTargets* renderTargets, const RendererConfig& config,
        const std::vector<GPUCommandSubmission*>& cmdToWait)
    {

        auto start = std::chrono::high_resolution_clock::now();
        PrepareImmutableResources();

        m_renderConfig     = config;
        m_config           = &config;
        auto& perframeData = *scene->GetPerFrameData();

        auto  frameId           = perframeData.m_frameId;
        auto  frameTimestampRaw = std::chrono::high_resolution_clock::now();
        auto  frameTimestampMicro =
            std::chrono::duration_cast<std::chrono::microseconds>(frameTimestampRaw.time_since_epoch());
        auto  frameTimestampMicroCount             = frameTimestampMicro.count();
        float frameTimestampMili                   = frameTimestampMicroCount / 1000.0f;
        perframeData.m_frameTimestamp[frameId % 2] = frameTimestampMili;

        auto haltonX = RendererConsts::cHalton2[frameId % RendererConsts::cHalton2.size()];
        auto haltonY = RendererConsts::cHalton3[frameId % RendererConsts::cHalton3.size()];

        u32  actualRw = 0, actualRh = 0;
        GetSupersampledRenderArea(renderTargets, &actualRw, &actualRh);
        auto               width       = actualRw;
        auto               height      = actualRh;
        auto               outputWidth = renderTargets->GetRenderArea().width;
        SceneCollectConfig sceneConfig;
        float              jx, jy;
        if (config.m_antiAliasingType == AntiAliasingType::TAA)
        {
            sceneConfig.projectionTranslateX = (haltonX * 2.0f - 1.0f) / width;
            sceneConfig.projectionTranslateY = (haltonY * 2.0f - 1.0f) / height;
        }
        else if (config.m_antiAliasingType == AntiAliasingType::FSR2)
        {

            m_fsr2proc->GetJitters(&jx, &jy, perframeData.m_frameId, actualRw, outputWidth);
            sceneConfig.projectionTranslateX = 2.0f * jx / width;

            // Note that we use Y Down in proj cam
            sceneConfig.projectionTranslateY = 2.0f * jy / height;
        }
        else
        {
            sceneConfig.projectionTranslateX = 0.0f;
            sceneConfig.projectionTranslateY = 0.0f;
        }

        // If debug views, ignore all jitters
        if (config.m_visualizationType != RendererVisualizationType::Default)
        {
            perframeData.m_taaJitterX        = 0;
            perframeData.m_taaJitterY        = 0;
            sceneConfig.projectionTranslateX = 0.0f;
            sceneConfig.projectionTranslateY = 0.0f;
        }
        SetRendererConfig(&config);
        CollectPerframeData(perframeData, scene, camera, GraphicsShaderPassType::Opaque, renderTargets, sceneConfig);

        auto end0      = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start);
        // iDebug("CPU time, frame collecting: {} ms", duration0.count());
        if (config.m_antiAliasingType == AntiAliasingType::TAA)
        {
            perframeData.m_taaJitterX = sceneConfig.projectionTranslateX * 0.5f;
            perframeData.m_taaJitterY = sceneConfig.projectionTranslateY * 0.5f;
        }
        else if (config.m_antiAliasingType == AntiAliasingType::FSR2)
        {
            perframeData.m_taaJitterX = jx;
            perframeData.m_taaJitterY = jy;
        }
        else
        {
            perframeData.m_taaJitterX = 0;
            perframeData.m_taaJitterY = 0;
        }

        // If debug views, ignore all jitters
        if (config.m_visualizationType != RendererVisualizationType::Default)
        {
            perframeData.m_taaJitterX        = 0;
            perframeData.m_taaJitterY        = 0;
            sceneConfig.projectionTranslateX = 0.0f;
            sceneConfig.projectionTranslateY = 0.0f;
        }

        auto ret = Render(perframeData, renderTargets, cmdToWait);

        if (m_renderRole & SyaroRenderRole::FullProcess)
        {
            perframeData.m_frameId++;
        }

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return ret;
    }

} // namespace Ifrit::Core