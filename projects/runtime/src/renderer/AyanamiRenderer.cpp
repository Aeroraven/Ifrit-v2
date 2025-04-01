
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

#include "ifrit/runtime/renderer/AyanamiRenderer.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"

#include "ifrit/core/math/constfunc/ConstFunc.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiSceneAggregator.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiTrivialSurfaceCache.h"

namespace Ifrit::Runtime
{
    using namespace Ifrit;
    using namespace Ifrit::Runtime::RenderingUtil;
    using namespace Ifrit::Runtime::Ayanami;

    struct AyanamiRendererResources
    {
        using DrawPass    = Graphics::Rhi::RhiGraphicsPass;
        using ComputePass = Graphics::Rhi::RhiComputePass;
        using GPUTexture  = Graphics::Rhi::RhiTextureRef;
        using GPUBindId   = Graphics::Rhi::RhiDescHandleLegacy;

        GPUTexture                              m_raymarchOutput = nullptr;
        Ref<GPUBindId>                          m_raymarchOutputSRVBindId;

        FrameGraphCompiler                      m_fgCompiler;
        FrameGraphExecutor                      m_fgExecutor;
        Uref<AyanamiSceneAggregator>            m_sceneAggregator;
        Uref<AyanamiTrivialSurfaceCacheManager> m_surfaceCacheManager = nullptr;

        ComputePass*                            m_raymarchPass = nullptr;
        DrawPass*                               m_debugPass    = nullptr;

        bool                                    m_inited          = false;
        bool                                    m_debugShowMeshDF = false;
    };

    IFRIT_APIDECL void AyanamiRenderer::InitRenderer()
    {
        m_resources                    = new AyanamiRendererResources();
        m_resources->m_sceneAggregator = std::make_unique<AyanamiSceneAggregator>(m_app->GetRhi());
        m_resources->m_surfaceCacheManager =
            std::make_unique<AyanamiTrivialSurfaceCacheManager>(m_selfRenderConfig, m_app);
    }
    IFRIT_APIDECL AyanamiRenderer::~AyanamiRenderer()
    {
        if (m_resources)
            delete m_resources;
    }

    IFRIT_APIDECL void AyanamiRenderer::PrepareResources(RenderTargets* renderTargets, const RendererConfig& config)
    {
        auto rhi    = m_app->GetRhi();
        auto width  = renderTargets->GetRenderArea().width;
        auto height = renderTargets->GetRenderArea().height;

        // Passes
        if (m_resources->m_debugPass == nullptr)
        {
            auto rtFmt               = renderTargets->GetFormat();
            m_resources->m_debugPass = CreateGraphicsPass(
                rhi, "Ayanami/Ayanami.CopyPass.vert.glsl", "Ayanami/Ayanami.CopyPass.frag.glsl", 0, 1, rtFmt);
        }
        if (m_resources->m_raymarchPass == nullptr)
        {
            m_resources->m_raymarchPass = CreateComputePass(rhi, "Ayanami/Ayanami.RayMarch.comp.glsl", 0, 6);
        }

        // Resources
        if (m_resources->m_raymarchOutput == nullptr)
        {
            using namespace Ifrit::Graphics::Rhi;
            auto sampler = m_immRes.m_linearSampler;
            m_resources->m_raymarchOutput =
                rhi->CreateTexture2D("Ayanami_Raymarch", width, height, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT, true);
            m_resources->m_raymarchOutputSRVBindId =
                rhi->RegisterCombinedImageSampler(m_resources->m_raymarchOutput.get(), sampler.get());
        }
    }

    IFRIT_APIDECL void AyanamiRenderer::SetupAndRunFrameGraph(
        PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        FrameGraphBuilder fg;
        fg.SetResourceInitState(FrameGraphResourceInitState::Uninitialized);

        auto rtWidth  = renderTargets->GetRenderArea().width;
        auto rtHeight = renderTargets->GetRenderArea().height;
        auto rhi      = m_app->GetRhi();

        if IF_CONSTEXPR (false)
            iWarn("Just here to make clang-format happy");

        auto& resSurfaceCacheAlbedo =
            fg.AddResource("SurfaceCacheAlbedo")
                .SetImportedResource(m_resources->m_surfaceCacheManager->GetAlbedoAtlas().get(), { 0, 0, 1, 1 });
        auto& resSurfaceCacheNormal =
            fg.AddResource("SurfaceCacheNormal")
                .SetImportedResource(m_resources->m_surfaceCacheManager->GetNormalAtlas().get(), { 0, 0, 1, 1 });
        auto& resSuraceCacheDepth =
            fg.AddResource("SurfaceCacheDepth")
                .SetImportedResource(m_resources->m_surfaceCacheManager->GetDepthAtlas().get(), { 0, 0, 1, 1 });

        auto& resRaymarchOutput =
            fg.AddResource("RaymarchOutput").SetImportedResource(m_resources->m_raymarchOutput.get(), { 0, 0, 1, 1 });

        auto& resGlobalDFGen =
            fg.AddResource("GlobalDFGen").SetImportedResource(m_globalDF->GetClipmapVolume(0).get(), { 0, 0, 1, 1 });

        auto& resRenderTargets =
            fg.AddResource("RenderTargets")
                .SetImportedResource(renderTargets->GetColorAttachment(0)->GetRenderTarget(), { 0, 0, 1, 1 });

        auto& passSurfaceCacheGen = fg.AddPass("SurfaceCacheGen", FrameGraphPassType::Graphics)
                                        .AddWriteResource(resSurfaceCacheAlbedo)
                                        .AddWriteResource(resSurfaceCacheNormal)
                                        .AddWriteResource(resSuraceCacheDepth);

        auto& passGlobalDFGen = fg.AddPass("GlobalDFGen", FrameGraphPassType::Compute);
        auto& passRaymarch    = fg.AddPass("RaymarchPass", FrameGraphPassType::Compute);
        auto& passDebug       = fg.AddPass("DebugPass", FrameGraphPassType::Graphics);

        if (!m_resources->m_inited)
        {
            passGlobalDFGen.AddWriteResource(resGlobalDFGen);

            passRaymarch.AddReadResource(resGlobalDFGen).AddWriteResource(resRaymarchOutput);

            passDebug.AddReadResource(resRaymarchOutput).AddWriteResource(resRenderTargets);
        }
        else
        {
            passRaymarch.AddReadResource(resGlobalDFGen).AddWriteResource(resRaymarchOutput);

            passDebug.AddReadResource(resRaymarchOutput).AddWriteResource(resRenderTargets);
        }

        if (!m_resources->m_inited)
        {
            passGlobalDFGen.SetExecutionFunction([&](const FrameGraphPassContext& data) {
                m_globalDF->AddClipmapUpdate(data.m_CmdList, 0, perframe.m_views[0].m_viewBufferId->GetActiveId(),
                    m_resources->m_sceneAggregator->GetNumGatheredInstances(),
                    m_resources->m_sceneAggregator->GetGatheredBufferId());
            });
        }
        else
        {
            passGlobalDFGen.SetExecutionFunction([&](const FrameGraphPassContext& data) {});
        }

        passSurfaceCacheGen.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto commandList = data.m_CmdList;
            m_resources->m_surfaceCacheManager->UpdateSurfaceCacheAtlas(commandList);
        });

        passRaymarch.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto commandList = data.m_CmdList;
            if (m_resources->m_debugShowMeshDF)
            {
                struct RayMarchPc
                {
                    u32 perframeId;
                    u32 totalInsts;
                    u32 descId;
                    u32 output;
                    u32 rtH;
                    u32 rtW;
                } pc;
                pc.rtH        = rtHeight;
                pc.rtW        = rtWidth;
                pc.totalInsts = m_resources->m_sceneAggregator->GetNumGatheredInstances();
                pc.output     = m_resources->m_raymarchOutput->GetDescId();
                pc.descId     = m_resources->m_sceneAggregator->GetGatheredBufferId();
                pc.perframeId = perframe.m_views[0].m_viewBufferId->GetActiveId();

                m_resources->m_raymarchPass->SetRecordFunction([&](const Graphics::Rhi::RhiRenderPassContext* ctx) {
                    commandList->SetPushConst(m_resources->m_raymarchPass, 0, sizeof(RayMarchPc), &pc);
                    commandList->Dispatch(Math::DivRoundUp(rtWidth, 8), Math::DivRoundUp(rtHeight, 8), 1);
                });
                m_resources->m_raymarchPass->Run(commandList, 0);
            }
            else
            {
                m_globalDF->AddRayMarchPass(commandList, 0, perframe.m_views[0].m_viewBufferId->GetActiveId(),
                    m_resources->m_raymarchOutput->GetDescId(), { rtWidth, rtHeight });
            }
        });

        passDebug.SetExecutionFunction([&](const FrameGraphPassContext& data) {
            auto commandList = data.m_CmdList;
            struct DispPc
            {
                u32 raymarchOutput;
            } pc;
            pc.raymarchOutput = m_resources->m_raymarchOutputSRVBindId->GetActiveId();
            EnqueueFullScreenPass(commandList, rhi, m_resources->m_debugPass, renderTargets, {}, &pc, 1);
        });

        auto compiledFg = m_resources->m_fgCompiler.Compile(fg);
        m_resources->m_fgExecutor.ExecuteInSingleCmd(cmd, compiledFg);
        m_resources->m_inited = true;
    }

    IFRIT_APIDECL Uref<AyanamiRenderer::GPUCommandSubmission> AyanamiRenderer::Render(Scene* scene, Camera* camera,
        RenderTargets* renderTargets, const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait)
    {

        m_config           = &config;
        auto& perframeData = *scene->GetPerFrameData();

        PrepareImmutableResources();
        PrepareResources(renderTargets, config);

        // Simply launch syaro's rendering for Gbuffer and shadowing
        // Now only directional light is considered, so no light culling is required
        auto vgTaskTimestamp = m_vgRenderer->Render(scene, camera, renderTargets, config, cmdToWait);

        // Start the main process
        m_resources->m_sceneAggregator->CollectScene(scene);
        m_resources->m_surfaceCacheManager->UpdateSceneCache(scene);
        auto rhi = m_app->GetRhi();
        auto dq  = rhi->GetQueue(Graphics::Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);

        auto task = dq->RunAsyncCommand(
            [&](const GPUCmdBuffer* cmd) { SetupAndRunFrameGraph(perframeData, renderTargets, cmd); },
            { vgTaskTimestamp.get() }, {});

        return task;
    }

} // namespace Ifrit::Runtime