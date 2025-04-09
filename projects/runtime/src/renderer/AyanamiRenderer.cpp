
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

#include "ifrit/core/math/constfunc/ConstFunc.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiSceneAggregator.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiTrivialSurfaceCache.h"

#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Ayanami.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiDFShadowing.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/renderer/ayanami//AyanamiDebugger.h"

using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Runtime::FrameGraphUtils;

namespace Ifrit::Runtime
{
    using namespace Ifrit;
    using namespace Ifrit::Runtime::RenderingUtil;
    using namespace Ifrit::Runtime::Ayanami;
    using namespace Ifrit::Graphics::Rhi;

    struct AyanamiRendererResources
    {
        using DrawPass    = RhiGraphicsPass;
        using ComputePass = RhiComputePass;
        using GPUTexture  = RhiTextureRef;
        using GPUBindId   = RhiDescHandleLegacy;
        using ColorRT     = RhiColorAttachment;
        using GPURT       = RhiRenderTargets;

        FrameGraphCompiler                      m_builderCompiler;
        Uref<FrameGraphExecutor>                m_builderExecutor;
        Uref<AyanamiSceneAggregator>            m_sceneAggregator;
        Uref<AyanamiTrivialSurfaceCacheManager> m_surfaceCacheManager = nullptr;
        Uref<AyanamiDistanceFieldLighting>      m_DFLighting          = nullptr;
        Uref<AyanamiDebugger>                   m_Debugger            = nullptr;

        bool                                    m_inited          = false;
        bool                                    m_debugShowMeshDF = false;

        Ref<FrameGraphResourcePool>             m_ResourcePool = nullptr;
    };

    static ComputePassNode& AddDFRadianceInjectPass(FrameGraphBuilder& builder, AyanamiRendererResources* res,
        Vector4f sceneBound, Vector3f lightDir, u32 cullTileSize, float softness)
    {

        auto& pass =
            res->m_DFLighting->AddDistanceFieldRadianceCachePass(builder, res->m_sceneAggregator->GetGatheredBufferId(),
                res->m_sceneAggregator->GetNumGatheredInstances(), &res->m_surfaceCacheManager->GetRDGDepthAtlas(),
                sceneBound, lightDir, &res->m_surfaceCacheManager->GetRDGShadowVisibilityAtlas(),
                res->m_surfaceCacheManager->GetCardDataBuffer()->GetDescId(),
                res->m_surfaceCacheManager->GetCardResolution(), res->m_surfaceCacheManager->GetCardAtlasResolution(),
                res->m_surfaceCacheManager->GetNumCards(), res->m_surfaceCacheManager->GetWorldMatsId(), cullTileSize,
                softness);
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
        m_resources->m_DFLighting      = std::make_unique<AyanamiDistanceFieldLighting>(m_app->GetRhi());
        m_resources->m_builderExecutor = std::make_unique<FrameGraphExecutor>(m_app->GetRhi());
        m_resources->m_Debugger        = std::make_unique<AyanamiDebugger>(m_app->GetRhi());
    }
    IFRIT_APIDECL AyanamiRenderer::~AyanamiRenderer()
    {
        if (m_resources)
            delete m_resources;
    }

    IFRIT_APIDECL void AyanamiRenderer::PrepareResources(RenderTargets* renderTargets, const RendererConfig& config) {}

    IFRIT_APIDECL void AyanamiRenderer::SetupAndRunFrameGraph(
        Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        FrameGraphBuilder builder(m_app->GetShaderRegistry(), m_app->GetRhi(), m_resources->m_ResourcePool.get());
        builder.SetResourceInitState(FrameGraphResourceInitState::Uninitialized);

        auto rtWidth  = renderTargets->GetRenderArea().width;
        auto rtHeight = renderTargets->GetRenderArea().height;
        auto rhi      = m_app->GetRhi();

        if IF_CONSTEXPR (false)
            iWarn("Just here to make clang-format happy");

        // Import resources

        auto& resGlobalDFGen = builder.ImportTexture("Ayanami.GlobalDFGen", m_globalDF->GetClipmapVolume(0).get());
        auto& resRenderTargets =
            builder.ImportTexture("Ayanami.RenderTargets", renderTargets->GetColorAttachment(0)->GetRenderTarget());
        auto& resGNormal = builder.ImportTexture("Ayanami.GBufferNormal", perframe.m_gbuffer.m_normal_smoothness.get());
        auto& resGDepth =
            builder.ImportTexture("Ayanami.GBufferDepth", perframe.m_views[0].m_visibilityDepth_Combined.get());

        // Managed resources
        auto& resRaymarchOutput = builder.DeclareTexture("Ayanami.RDG.RayMarchOutput",
            FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_ShaderRead));
        auto& resDfssOut        = builder.DeclareTexture("Ayanami.RDG.DFSSOutput",
                   FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                       RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead));
        auto& resDeferOut       = builder.DeclareTexture("Ayanami.RDG.DeferShadingOut",
                  FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                      RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead));
        auto& resDebugSCOut     = builder.DeclareTexture("Ayanami.RDG.DebugSCOut",
                FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                    RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead
                        | RhiImageUsage::RhiImgUsage_UnorderedAccess));

        m_resources->m_surfaceCacheManager->InitContext(builder);
        m_resources->m_DFLighting->InitContext(builder, 64);

        // Pass Global DF Generation
        if (!m_resources->m_inited)
        {
            m_globalDF
                ->AddClipmapUpdate(builder, 0, perframe.m_views[0].m_viewBufferId->GetActiveId(),
                    m_resources->m_sceneAggregator->GetNumGatheredInstances(),
                    m_resources->m_sceneAggregator->GetGatheredBufferId())
                .AddWriteResource(resGlobalDFGen);
        }
        // Pass Surface Cache + Radiance Injection (Camera View)
        if (!m_resources->m_inited || m_selfRenderConfig.m_DebugForceSurfaceCacheRegen)
        {
            m_resources->m_surfaceCacheManager->UpdateSurfaceCacheAtlas(builder);
        }
        m_resources->m_surfaceCacheManager->UpdateRadianceCacheAtlas(builder, scene);

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
        m_resources->m_DFLighting->DistanceFieldShadowTileScatter(builder,
            m_resources->m_sceneAggregator->GetGatheredBufferId(),
            m_resources->m_sceneAggregator->GetNumGatheredInstances(), sceneBound, sceneLight, 64);

        // printf("Scene bound: %f %f %f %f\n", sceneBound.x, sceneBound.y, sceneBound.z, sceneBound.w);

        // Pass DF Radiance Injection (World Space)
        AddDFRadianceInjectPass(builder, m_resources, sceneBound, sceneLight, 64, 2.0f);

        // Pass Direct Lighting
        m_resources->m_surfaceCacheManager->UpdateDirectLighting(
            builder, m_resources->m_sceneAggregator->GetGatheredBufferId(), sceneLight);

        // Pass Voxel Construction (Object Grids)
        m_globalDF->AddObjectGridCompositionPass(builder, 0, m_resources->m_sceneAggregator->GetNumGatheredInstances(),
            m_resources->m_sceneAggregator->GetGatheredBufferId());

        // Pass Indirect Radiance
        auto globalDFSrvId = m_globalDF->GetClipmapVolumeSRV(0);
        m_resources->m_surfaceCacheManager
            ->UpdateIndirectRadianceCacheAtlas(
                builder, scene, globalDFSrvId, m_resources->m_sceneAggregator->GetGatheredBufferId())
            .AddReadResource(resGlobalDFGen);

        // Pass RayMarch
        if (m_resources->m_debugShowMeshDF)
        {
            struct PushConst
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
            pc.descId     = m_resources->m_sceneAggregator->GetGatheredBufferId();
            pc.perframeId = perframe.m_views[0].m_viewBufferId->GetActiveId();

            AddComputePass<PushConst>(builder, "Ayanami.RaymarchPass", Internal::kIntShaderTableAyanami.RayMarchCS,
                Vector3i{ Math::DivRoundUp<i32>(rtWidth, 8), Math::DivRoundUp<i32>(rtHeight, 8), 1 }, pc,
                [&resRaymarchOutput](PushConst data, const FrameGraphPassContext& ctx) {
                    data.output = ctx.m_FgDesc->GetUAV(resRaymarchOutput);
                    SetRootSignature(data, ctx);
                })
                .AddReadResource(resGlobalDFGen)
                .AddWriteResource(resRaymarchOutput);
        }
        else
        {
            m_globalDF
                ->AddRayMarchPass(builder, 0, perframe.m_views[0].m_viewBufferId->GetActiveId(), &resRaymarchOutput,
                    { rtWidth, rtHeight })
                .AddReadResource(resGlobalDFGen)
                .AddWriteResource(resRaymarchOutput);
        }
        // Pass DFSS
        m_resources->m_DFLighting
            ->DistanceFieldShadowRender(builder, m_resources->m_sceneAggregator->GetGatheredBufferId(),
                m_resources->m_sceneAggregator->GetNumGatheredInstances(),
                perframe.m_views[0].m_visibilityDepthIdSRV_Combined->GetActiveId(),
                perframe.m_views[0].m_viewBufferId->GetActiveId(), sceneBound, sceneLight, 64, 2)
            .AddRenderTarget(resDfssOut)
            .AddReadResource(resGDepth);

        // Pass Defered Shading
        {
            auto& resSurfaceCacheNormal = m_resources->m_surfaceCacheManager->GetRDGNormalAtlas();
            struct PushConst
            {
                Vector4f lightDir;
                u32      normalSRV;
                u32      perframeId;
                u32      m_ShadowMapSRV = 0;
            } pc;
            pc.normalSRV  = perframe.m_gbuffer.m_normal_smoothness_sampId->GetActiveId();
            pc.perframeId = perframe.m_views[0].m_viewBufferId->GetActiveId();
            pc.lightDir   = Vector4f(sceneLights.m_LightFronts[0], 0.0f);
            AddPostProcessPass<PushConst>(builder, "Ayanami.DeferredShading",
                Internal::kIntShaderTableAyanami.TestDeferShadingFS, pc,
                [&resDfssOut](PushConst data, const FrameGraphPassContext& ctx) {
                    data.m_ShadowMapSRV = ctx.m_FgDesc->GetSRV(resDfssOut);
                    SetRootSignature(data, ctx);
                })
                .AddRenderTarget(resDeferOut)
                .AddReadResource(resDfssOut)
                .AddReadResource(resSurfaceCacheNormal);
        }
        // Pass Surface Cache Debug
        {
            auto& resAlbedoAtlas   = m_resources->m_surfaceCacheManager->GetRDGAlbedoAtlas();
            auto& resNormalAtlas   = m_resources->m_surfaceCacheManager->GetRDGNormalAtlas();
            auto& resRadianceAtlas = m_resources->m_surfaceCacheManager->GetRDGShadowVisibilityAtlas();
            auto& resDepthAtlas    = m_resources->m_surfaceCacheManager->GetRDGDepthAtlas();

            m_resources->m_Debugger->RenderSceneFromCacheSurface(builder, &resDebugSCOut, &resAlbedoAtlas,
                &resNormalAtlas, &resRadianceAtlas, &resDepthAtlas, m_resources->m_surfaceCacheManager->GetNumCards(),
                m_resources->m_surfaceCacheManager->GetCardResolution(),
                m_resources->m_surfaceCacheManager->GetCardAtlasResolution(),
                m_resources->m_surfaceCacheManager->GetCardDataBuffer()->GetDescId(),
                perframe.m_views[0].m_viewBufferId->GetActiveId(),
                m_resources->m_sceneAggregator->GetGatheredBufferId());
        }

        // Pass Debug
        {
            auto& resDirectRadiance = m_resources->m_surfaceCacheManager->GetRDGShadowVisibilityAtlas();
            struct PushConst
            {
                u32 raymarchOutput = 0;
            } pc;
            AddFullScreenQuadPass<PushConst>(builder, "Ayanami.DebugPass", Internal::kIntShaderTableAyanami.CopyVS,
                Internal::kIntShaderTableAyanami.CopyFS, pc,
                [&resRaymarchOutput](PushConst data, const FrameGraphPassContext& ctx) {
                    data.raymarchOutput = ctx.m_FgDesc->GetSRV(resRaymarchOutput);
                    SetRootSignature(data, ctx);
                })
                .AddRenderTarget(resRenderTargets)
                .AddReadResource(resDebugSCOut)
                .AddReadResource(resRaymarchOutput)
                .AddReadResource(resDirectRadiance)
                .AddReadResource(resDeferOut)
                .AddReadResource(resGNormal);
        }

        auto compiledFg = m_resources->m_builderCompiler.Compile(builder);
        m_resources->m_builderExecutor->ExecuteInSingleCmd(cmd, compiledFg);
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
        auto dq  = rhi->GetQueue(RhiQueueCapability::RhiQueue_Graphics);

        auto task = dq->RunAsyncCommand(
            [&](const GPUCmdBuffer* cmd) {
                cmd->BeginScope("Ayanami: Execute Render Graph");
                SetupAndRunFrameGraph(scene, perframeData, renderTargets, cmd);
                cmd->EndScope();
            },
            { vgTaskTimestamp.get() }, {});

        return task;
    }

} // namespace Ifrit::Runtime