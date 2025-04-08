
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
    using namespace Ifrit::Graphics::Rhi;

    struct AyanamiRendererResources
    {
        using DrawPass    = Graphics::Rhi::RhiGraphicsPass;
        using ComputePass = Graphics::Rhi::RhiComputePass;
        using GPUTexture  = Graphics::Rhi::RhiTextureRef;
        using GPUBindId   = Graphics::Rhi::RhiDescHandleLegacy;
        using ColorRT     = Graphics::Rhi::RhiColorAttachment;
        using GPURT       = Graphics::Rhi::RhiRenderTargets;

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

        GPUTexture                              m_DeferShadingOut    = nullptr;
        Ref<GPUBindId>                          m_DeferShadingOutSRV = nullptr;
        Ref<ColorRT>                            m_DeferShadingOutRT  = nullptr;
        Ref<GPURT>                              m_DeferShadingOutRTs = nullptr;

        GPUTexture                              m_DfssOut          = nullptr;
        Ref<GPUBindId>                          m_DfssOutSRVBindId = nullptr;
        Ref<ColorRT>                            m_DfssOutRT        = nullptr;
        Ref<GPURT>                              m_DfssOutRTs       = nullptr;

        Ref<FrameGraphResourcePool>             m_ResourcePool = nullptr;
    };

    static ComputePassNode& AddDFRadianceInjectPass(FrameGraphBuilder& builder, AyanamiRendererResources* res,
        Vector4f sceneBound, Vector3f lightDir, u32 cullTileSize, float softness, ResourceNode& radianceAtlas,
        ResourceNode& depthAtlas)
    {

        auto& pass =
            res->m_DFLighting->AddDistanceFieldRadianceCachePass(builder, res->m_sceneAggregator->GetGatheredBufferId(),
                res->m_sceneAggregator->GetNumGatheredInstances(), res->m_surfaceCacheManager->GetDepthSRVId(),
                sceneBound, lightDir, res->m_surfaceCacheManager->GetRadianceAtlas()->GetDescId(),
                res->m_surfaceCacheManager->GetCardDataBuffer()->GetDescId(),
                res->m_surfaceCacheManager->GetCardResolution(), res->m_surfaceCacheManager->GetCardAtlasResolution(),
                res->m_surfaceCacheManager->GetNumCards(), res->m_surfaceCacheManager->GetWorldMatsId(), cullTileSize,
                softness);
        pass.AddReadWriteResource(radianceAtlas).AddReadResource(depthAtlas);
        return pass;
    }

    IFRIT_APIDECL void AyanamiRenderer::InitRenderer()
    {
        m_resources                 = new AyanamiRendererResources();
        m_resources->m_ResourcePool = std::make_shared<FrameGraphResourcePool>(m_app->GetRhi());
        m_resources->m_sceneAggregator =
            std::make_unique<AyanamiSceneAggregator>(m_app->GetRhi(), m_app->GetSharedRenderResource());
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
            auto sampler = m_app->GetSharedRenderResource()->GetLinearClampSampler();
            m_resources->m_raymarchOutput =
                rhi->CreateTexture2D("Ayanami_Raymarch", width, height, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_ShaderRead, true);
            m_resources->m_raymarchOutputSRVBindId =
                rhi->RegisterCombinedImageSampler(m_resources->m_raymarchOutput.get(), sampler.get());
        }

        if (m_resources->m_DeferShadingOut == nullptr)
        {
            m_resources->m_DeferShadingOut   = rhi->CreateTexture2D("Ayanami_DeferShadingOut", width, height,
                  RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                  RhiImageUsage::RhiImgUsage_RenderTarget | RhiImgUsage_ShaderRead, false);
            m_resources->m_DeferShadingOutRT = rhi->CreateRenderTarget(
                m_resources->m_DeferShadingOut.get(), { 0, 0, 1, 1 }, RhiRenderTargetLoadOp::Clear, 0, 0);
            m_resources->m_DeferShadingOutRTs = rhi->CreateRenderTargets();
            m_resources->m_DeferShadingOutRTs->SetColorAttachments({ m_resources->m_DeferShadingOutRT.get() });
            m_resources->m_DeferShadingOutRTs->SetRenderArea({ 0, 0, width, height });

            m_resources->m_DeferShadingOutSRV = rhi->RegisterCombinedImageSampler(
                m_resources->m_DeferShadingOut.get(), m_app->GetSharedRenderResource()->GetLinearRepeatSampler().get());
        }

        if (m_resources->m_DfssOut == nullptr)
        {
            m_resources->m_DfssOut =
                rhi->CreateTexture2D("Ayanami_DFSSOut", width, height, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RhiImgUsage_RenderTarget | RhiImgUsage_ShaderRead, false);
            m_resources->m_DfssOutRT = rhi->CreateRenderTarget(
                m_resources->m_DfssOut.get(), { 0, 0, 1, 1 }, RhiRenderTargetLoadOp::Clear, 0, 0);
            m_resources->m_DfssOutRTs = rhi->CreateRenderTargets();
            m_resources->m_DfssOutRTs->SetColorAttachments({ m_resources->m_DfssOutRT.get() });
            m_resources->m_DfssOutRTs->SetRenderArea({ 0, 0, width, height });

            m_resources->m_DfssOutSRVBindId = rhi->RegisterCombinedImageSampler(
                m_resources->m_DfssOut.get(), m_app->GetSharedRenderResource()->GetLinearRepeatSampler().get());
        }
    }

    IFRIT_APIDECL void AyanamiRenderer::SetupAndRunFrameGraph(
        Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        FrameGraphBuilder fg(m_app->GetShaderRegistry(), m_app->GetRhi(), m_resources->m_ResourcePool.get());
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
        auto& resTracedRadiance =
            fg.AddResource("TracedRadiance")
                .SetImportedResource(
                    m_resources->m_surfaceCacheManager->GetTracedRadianceAtlas().get(), { 0, 0, 1, 1 });

        auto& resRaymarchOutput =
            fg.AddResource("RaymarchOutput").SetImportedResource(m_resources->m_raymarchOutput.get(), { 0, 0, 1, 1 });
        auto& resGlobalDFGen =
            fg.AddResource("GlobalDFGen").SetImportedResource(m_globalDF->GetClipmapVolume(0).get(), { 0, 0, 1, 1 });
        auto& resRenderTargets =
            fg.AddResource("RenderTargets")
                .SetImportedResource(renderTargets->GetColorAttachment(0)->GetRenderTarget(), { 0, 0, 1, 1 });
        auto& resDeferOut =
            fg.AddResource("DeferShadingOut").SetImportedResource(m_resources->m_DeferShadingOut.get(), { 0, 0, 1, 1 });
        auto& resGNormal = fg.AddResource("GBufferNormal")
                               .SetImportedResource(perframe.m_gbuffer.m_normal_smoothness.get(), { 0, 0, 1, 1 });
        auto& resGDepth =
            fg.AddResource("GBufferDepth")
                .SetImportedResource(perframe.m_views[0].m_visibilityDepth_Combined.get(), { 0, 0, 1, 1 });
        auto& resDfssOut = fg.AddResource("DfssOut").SetImportedResource(m_resources->m_DfssOut.get(), { 0, 0, 1, 1 });

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
        if (!m_resources->m_inited)
        {
            // todo: dynamic update
            m_resources->m_surfaceCacheManager->UpdateSurfaceCacheAtlas(fg)
                .AddWriteResource(resSurfaceCacheAlbedo)
                .AddWriteResource(resSurfaceCacheNormal)
                .AddWriteResource(resSuraceCacheDepth);
        }

        m_resources->m_surfaceCacheManager->UpdateRadianceCacheAtlas(fg, scene).AddWriteResource(resDirectRadiance);

        // Pass DF Culling
        auto sceneBound  = m_resources->m_sceneAggregator->GetSceneBoundSphere();
        auto sceneLights = m_resources->m_sceneAggregator->GetAggregatedLights();
        if (sceneLights.m_LightFronts.size() != 1)
        {
            iError("AyanamiRenderer: temporarily only support one light for now, got {}",
                sceneLights.m_LightFronts.size());
            std::abort();
        }
        auto sceneLight = sceneLights.m_LightFronts[0];
        m_resources->m_DFLighting->DistanceFieldShadowTileScatter(fg,
            m_resources->m_sceneAggregator->GetGatheredBufferId(),
            m_resources->m_sceneAggregator->GetNumGatheredInstances(), sceneBound, sceneLight, 64);

        // Pass DF Radiance Injection (World Space)
        AddDFRadianceInjectPass(
            fg, m_resources, sceneBound, sceneLight, 64, 2.0f, resDirectRadiance, resSurfaceCacheNormal);

        // Pass Voxel Construction (Object Grids)
        m_globalDF->AddObjectGridCompositionPass(fg, 0, m_resources->m_sceneAggregator->GetNumGatheredInstances(),
            m_resources->m_sceneAggregator->GetGatheredBufferId());

        // Pass Indirect Radiance
        auto globalDFSrvId = m_globalDF->GetClipmapVolumeSRV(0);
        m_resources->m_surfaceCacheManager
            ->UpdateIndirectRadianceCacheAtlas(
                fg, scene, globalDFSrvId, m_resources->m_sceneAggregator->GetGatheredBufferId())
            .AddWriteResource(resTracedRadiance)
            .AddReadResource(resGlobalDFGen)
            .AddReadResource(resSurfaceCacheNormal)
            .AddReadResource(resSuraceCacheDepth);

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

            FrameGraphUtils::AddComputePass(fg, "Ayanami.RaymarchPass", Internal::kIntShaderTable.Ayanami.RayMarchCS,
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
        // Pass DFSS
        m_resources->m_DFLighting
            ->DistanceFieldShadowRender(fg, m_resources->m_sceneAggregator->GetGatheredBufferId(),
                m_resources->m_sceneAggregator->GetNumGatheredInstances(),
                perframe.m_views[0].m_visibilityDepthIdSRV_Combined->GetActiveId(),
                perframe.m_views[0].m_viewBufferId->GetActiveId(), m_resources->m_DfssOutRTs.get(), sceneBound,
                sceneLight, 64, 2)
            .AddWriteResource(resDfssOut)
            .AddReadResource(resGDepth);

        // Pass Defered Shading
        {
            struct DeferShadingPc
            {
                Vector4f lightDir;
                u32      normalSRV;
                u32      perframeId;
                u32      shadowMapSRV;
            } pc;
            pc.normalSRV    = perframe.m_gbuffer.m_normal_smoothness_sampId->GetActiveId();
            pc.perframeId   = perframe.m_views[0].m_viewBufferId->GetActiveId();
            auto l          = sceneLights.m_LightFronts[0];
            pc.lightDir     = Vector4f{ l.x, l.y, l.z, 0.0f };
            pc.shadowMapSRV = m_resources->m_DfssOutSRVBindId->GetActiveId();
            FrameGraphUtils::AddPostProcessPass(fg, "Ayanami.DeferredShading",
                Internal::kIntShaderTable.Ayanami.TestDeferShadingFS, m_resources->m_DeferShadingOutRTs.get(), &pc,
                FrameGraphUtils::GetPushConstSize<DeferShadingPc>())
                .AddWriteResource(resDeferOut)
                .AddReadResource(resDfssOut)
                .AddReadResource(resSurfaceCacheNormal);
        }

        // Pass Debug
        {
            struct DebugPassPc
            {
                u32 raymarchOutput;
            } pc;
            pc.raymarchOutput = m_resources->m_DeferShadingOutSRV->GetActiveId();
            // pc.raymarchOutput = m_resources->m_surfaceCacheManager->GetRadianceSRVId();
            FrameGraphUtils::AddFullScreenQuadPass(fg, "Ayanami.DebugPass", Internal::kIntShaderTable.Ayanami.CopyVS,
                Internal::kIntShaderTable.Ayanami.CopyFS, renderTargets, &pc, 1)
                .AddReadResource(resRaymarchOutput)
                .AddReadResource(resDirectRadiance)
                .AddReadResource(resDeferOut)
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
        auto dq  = rhi->GetQueue(Graphics::Rhi::RhiQueueCapability::RhiQueue_Graphics);

        auto task = dq->RunAsyncCommand(
            [&](const GPUCmdBuffer* cmd) { SetupAndRunFrameGraph(scene, perframeData, renderTargets, cmd); },
            { vgTaskTimestamp.get() }, {});

        return task;
    }

} // namespace Ifrit::Runtime