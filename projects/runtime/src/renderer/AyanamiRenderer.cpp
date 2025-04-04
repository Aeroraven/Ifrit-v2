
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

#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiDFShadowing.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"

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
        Uref<AyanamiDistanceFieldLighting>      m_DFLighting          = nullptr;

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
        m_resources->m_DFLighting = std::make_unique<AyanamiDistanceFieldLighting>(m_app->GetRhi());
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
            m_resources->m_debugPass = CreateGraphicsPassInternal(
                m_app, Internal::kIntShaderTable.Ayanami.CopyVS, Internal::kIntShaderTable.Ayanami.CopyFS, 0, 1, rtFmt);
        }
        if (m_resources->m_raymarchPass == nullptr)
        {
            m_resources->m_raymarchPass =
                CreateComputePassInternal(m_app, Internal::kIntShaderTable.Ayanami.RayMarchCS, 0, 6);
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
        Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        FrameGraphBuilder fg(m_app->GetShaderRegistry(), m_app->GetRhi());
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
        auto& resDirectRadiance =
            fg.AddResource("DirectRadiance")
                .SetImportedResource(m_resources->m_surfaceCacheManager->GetRadianceAtlas().get(), { 0, 0, 1, 1 });

        auto& resRaymarchOutput =
            fg.AddResource("RaymarchOutput").SetImportedResource(m_resources->m_raymarchOutput.get(), { 0, 0, 1, 1 });

        auto& resGlobalDFGen =
            fg.AddResource("GlobalDFGen").SetImportedResource(m_globalDF->GetClipmapVolume(0).get(), { 0, 0, 1, 1 });

        auto& resRenderTargets =
            fg.AddResource("RenderTargets")
                .SetImportedResource(renderTargets->GetColorAttachment(0)->GetRenderTarget(), { 0, 0, 1, 1 });

        // Pass Global DF Generation
        if (!m_resources->m_inited)
        {
            m_globalDF
                ->AddClipmapUpdate(fg, 0, perframe.m_views[0].m_viewBufferId->GetActiveId(),
                    m_resources->m_sceneAggregator->GetNumGatheredInstances(),
                    m_resources->m_sceneAggregator->GetGatheredBufferId())
                .AddWriteResource(resGlobalDFGen);
        }
        // Pass Surface Cache + Radiance Injection (Camera View)
        m_resources->m_surfaceCacheManager->UpdateSurfaceCacheAtlas(fg)
            .AddWriteResource(resSurfaceCacheAlbedo)
            .AddWriteResource(resSurfaceCacheNormal)
            .AddWriteResource(resSuraceCacheDepth);

        m_resources->m_surfaceCacheManager->UpdateRadianceCacheAtlas(fg, scene).AddWriteResource(resDirectRadiance);

        // Pass DF Culling (Incomplete)
        m_resources->m_DFLighting->DistanceFieldShadowTileScatter(fg,
            m_resources->m_sceneAggregator->GetGatheredBufferId(),
            m_resources->m_sceneAggregator->GetNumGatheredInstances(), Vector4f(0, 0, 0, 24.0f), Vector3f(0, 0, -1),
            64);

        // Pass RayMarch
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

            FrameGraphUtils::AddComputePass(fg, "Ayanami/RaymarchPass", Internal::kIntShaderTable.Ayanami.RayMarchCS,
                Vector3i{ Math::DivRoundUp<i32>(rtWidth, 8), Math::DivRoundUp<i32>(rtHeight, 8), 1 }, &pc, 6)
                .AddReadResource(resGlobalDFGen)
                .AddWriteResource(resRaymarchOutput);
        }
        else
        {
            m_globalDF
                ->AddRayMarchPass(fg, 0, perframe.m_views[0].m_viewBufferId->GetActiveId(),
                    m_resources->m_raymarchOutput->GetDescId(), { rtWidth, rtHeight })
                .AddReadResource(resGlobalDFGen)
                .AddWriteResource(resRaymarchOutput);
        }

        // Pass Debug
        {
            struct DebugPassPc
            {
                u32 raymarchOutput;
            } pc;
            pc.raymarchOutput = m_resources->m_surfaceCacheManager->GetRadianceSRVId();
            FrameGraphUtils::AddFullScreenQuadPass(fg, "Ayanami/DebugPass", Internal::kIntShaderTable.Ayanami.CopyVS,
                Internal::kIntShaderTable.Ayanami.CopyFS, renderTargets, &pc, 1)
                .AddReadResource(resRaymarchOutput)
                .AddReadResource(resDirectRadiance)
                .AddWriteResource(resRenderTargets);
        }

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
            [&](const GPUCmdBuffer* cmd) { SetupAndRunFrameGraph(scene, perframeData, renderTargets, cmd); },
            { vgTaskTimestamp.get() }, {});

        return task;
    }

} // namespace Ifrit::Runtime