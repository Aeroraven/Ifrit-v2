
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

        FrameGraphCompiler                      m_FgCompiler;
        Uref<FrameGraphExecutor>                m_FgExecutor;
        Uref<AyanamiSceneAggregator>            m_SceneAggregator;
        Uref<AyanamiTrivialSurfaceCacheManager> m_SurfaceCache = nullptr;
        Uref<AyanamiDistanceFieldLighting>      m_DFLighting   = nullptr;
        Uref<AyanamiDebugger>                   m_Debugger     = nullptr;

        bool                                    m_Inited     = false;
        bool                                    m_DbgShowMDF = false;

        Ref<FrameGraphResourcePool>             m_ResourcePool = nullptr;
    };

    static ComputePassNode& AddDFRadianceInjectPass(FrameGraphBuilder& builder, AyanamiRendererResources* res,
        Vector4f sceneBound, Vector3f lightDir, u32 cullTileSize, float softness)
    {

        auto& pass = res->m_DFLighting->AddDistanceFieldRadianceCachePass(builder,
            res->m_SceneAggregator->GetGatheredBufferId(), res->m_SceneAggregator->GetNumGatheredInstances(),
            &res->m_SurfaceCache->GetRDGDepthAtlas(), sceneBound, lightDir,
            &res->m_SurfaceCache->GetRDGShadowVisibilityAtlas(), res->m_SurfaceCache->GetCardDataBuffer()->GetDescId(),
            res->m_SurfaceCache->GetCardResolution(), res->m_SurfaceCache->GetCardAtlasResolution(),
            res->m_SurfaceCache->GetNumCards(), res->m_SurfaceCache->GetWorldMatsId(), cullTileSize, softness);
        return pass;
    }

    IFRIT_APIDECL void AyanamiRenderer::InitRenderer()
    {
        m_resources                 = new AyanamiRendererResources();
        m_resources->m_ResourcePool = std::make_shared<FrameGraphResourcePool>(m_app->GetRhi());
        m_resources->m_SceneAggregator =
            std::make_unique<AyanamiSceneAggregator>(m_app->GetRhi(), m_app->GetSharedRenderResource());
        m_resources->m_SurfaceCache = std::make_unique<AyanamiTrivialSurfaceCacheManager>(m_selfRenderConfig, m_app);
        m_resources->m_DFLighting   = std::make_unique<AyanamiDistanceFieldLighting>(m_app->GetRhi());
        m_resources->m_FgExecutor   = std::make_unique<FrameGraphExecutor>(m_app->GetRhi());
        m_resources->m_Debugger     = std::make_unique<AyanamiDebugger>(m_app->GetRhi());
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

        // Init
        m_resources->m_SurfaceCache->InitContext(builder);
        m_resources->m_DFLighting->InitContext(builder, 64);
        m_globalDF->InitContext(builder);

        // Import resources
        auto& resRenderTargets =
            builder.ImportTexture("Ayanami.RenderTargets", renderTargets->GetColorAttachment(0)->GetRenderTarget());
        auto& resGNormal = builder.ImportTexture("Ayanami.GBufferNormal", perframe.m_gbuffer.m_normal_smoothness.get());
        auto& resGDepth =
            builder.ImportTexture("Ayanami.GBufferDepth", perframe.m_views[0].m_visibilityDepth_Combined.get());
        auto& resGlobalDFGen      = *m_globalDF->GetClipmapVolume(0);
        auto& resGlobalObjectGrid = *m_globalDF->GetObjectGridVolume(0);

        // Managed resources
        auto& resRaymarchOutput  = builder.DeclareTexture("Ayanami.RDG.RayMarchOutput",
             FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                 RhiImageUsage::RhiImgUsage_UnorderedAccess | RhiImageUsage::RhiImgUsage_ShaderRead));
        auto& resDfssOut         = builder.DeclareTexture("Ayanami.RDG.DFSSOutput",
                    FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                        RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead));
        auto& resDeferOut        = builder.DeclareTexture("Ayanami.RDG.DeferShadingOut",
                   FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                       RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead));
        auto& resDebugSCOut      = builder.DeclareTexture("Ayanami.RDG.DebugSCOut",
                 FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                     RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead
                         | RhiImageUsage::RhiImgUsage_UnorderedAccess));
        auto& resDebugObjGridOut = builder.DeclareTexture("Ayanami.RDG.DebugObjGridOut",
            FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead
                    | RhiImageUsage::RhiImgUsage_UnorderedAccess));
        auto& resDebugObjGridVis = builder.DeclareTexture("Ayanami.RDG.DebugObjGridVis",
            FrameGraphTextureDesc(rtWidth, rtHeight, 1, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_ShaderRead
                    | RhiImageUsage::RhiImgUsage_UnorderedAccess));

        // Shared
        auto  primaryViewCBV = perframe.m_views[0].m_viewBufferId->GetActiveId();

        // Pass Global DF Generation
        if (!m_resources->m_Inited)
        {
            m_globalDF
                ->AddClipmapUpdate(builder, 0, primaryViewCBV,
                    m_resources->m_SceneAggregator->GetNumGatheredInstances(),
                    m_resources->m_SceneAggregator->GetGatheredBufferId())
                .AddWriteResource(resGlobalDFGen);
        }
        // Pass Surface Cache + Radiance Injection (Camera View)
        if (!m_resources->m_Inited || m_selfRenderConfig.m_DebugForceSurfaceCacheRegen)
        {
            m_resources->m_SurfaceCache->UpdateSurfaceCacheAtlas(builder);
        }
        m_resources->m_SurfaceCache->UpdateShadowVisibilityAtlas(builder, scene);

        // Pass DF Culling
        auto sceneBound  = m_resources->m_SceneAggregator->GetSceneBoundSphere();
        auto sceneLights = m_resources->m_SceneAggregator->GetAggregatedLights();
        if (sceneLights.m_LightFronts.size() != 1)
        {
            iError("AyanamiRenderer: temporarily only support one light for now, got {}",
                sceneLights.m_LightFronts.size());
            std::abort();
        }
        auto sceneLight = sceneLights.m_LightFronts[0];
        m_resources->m_DFLighting->DistanceFieldShadowTileScatter(builder,
            m_resources->m_SceneAggregator->GetGatheredBufferId(),
            m_resources->m_SceneAggregator->GetNumGatheredInstances(), sceneBound, sceneLight, 64);

        // printf("Scene bound: %f %f %f %f\n", sceneBound.x, sceneBound.y, sceneBound.z, sceneBound.w);

        // Pass DF Radiance Injection (World Space)
        AddDFRadianceInjectPass(builder, m_resources, sceneBound, sceneLight, 64, 2.0f);

        // Pass Direct Lighting
        m_resources->m_SurfaceCache->UpdateDirectLighting(
            builder, m_resources->m_SceneAggregator->GetGatheredBufferId(), sceneLight);

        // Pass Voxel Construction (Object Grids)
        m_globalDF->AddObjectGridCompositionPass(builder, 0, m_resources->m_SceneAggregator->GetNumGatheredInstances(),
            m_resources->m_SceneAggregator->GetGatheredBufferId());

        // Pass Indirect Radiance
        m_resources->m_SurfaceCache->UpdateIndirectRadianceCacheAtlas(
            builder, scene, &resGlobalDFGen, m_resources->m_SceneAggregator->GetGatheredBufferId());

        // Pass RayMarch
        if (true)
        {
            if (m_resources->m_DbgShowMDF)
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
                pc.totalInsts = m_resources->m_SceneAggregator->GetNumGatheredInstances();
                pc.descId     = m_resources->m_SceneAggregator->GetGatheredBufferId();
                pc.perframeId = primaryViewCBV;

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
                m_globalDF->AddRayMarchPass(builder, 0, primaryViewCBV, &resRaymarchOutput, { rtWidth, rtHeight })
                    .AddReadResource(resGlobalDFGen)
                    .AddWriteResource(resRaymarchOutput);
            }
        }

        // Pass DFSS
        if (false)
        {
            m_resources->m_DFLighting
                ->DistanceFieldShadowRender(builder, m_resources->m_SceneAggregator->GetGatheredBufferId(),
                    m_resources->m_SceneAggregator->GetNumGatheredInstances(),
                    perframe.m_views[0].m_visibilityDepthIdSRV_Combined->GetActiveId(), primaryViewCBV, sceneBound,
                    sceneLight, 64, 2)
                .AddRenderTarget(resDfssOut)
                .AddReadResource(resGDepth);
        }

        // Pass Defered Shading

        {
            auto& resSurfaceCacheNormal = m_resources->m_SurfaceCache->GetRDGNormalAtlas();
            struct PushConst
            {
                Vector4f lightDir;
                u32      normalSRV;
                u32      perframeId;
                u32      m_ShadowMapSRV = 0;
            } pc;
            pc.normalSRV  = perframe.m_gbuffer.m_normal_smoothness_sampId->GetActiveId();
            pc.perframeId = primaryViewCBV;
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
        if (false)
        {
            auto& resAlbedoAtlas   = m_resources->m_SurfaceCache->GetRDGAlbedoAtlas();
            auto& resNormalAtlas   = m_resources->m_SurfaceCache->GetRDGNormalAtlas();
            auto& resRadianceAtlas = m_resources->m_SurfaceCache->GetRDGShadowVisibilityAtlas();
            auto& resDepthAtlas    = m_resources->m_SurfaceCache->GetRDGDepthAtlas();

            m_resources->m_Debugger->RenderSceneFromCacheSurface(builder, &resDebugSCOut, &resAlbedoAtlas,
                &resNormalAtlas, &resRadianceAtlas, &resDepthAtlas, m_resources->m_SurfaceCache->GetNumCards(),
                m_resources->m_SurfaceCache->GetCardResolution(), m_resources->m_SurfaceCache->GetCardAtlasResolution(),
                m_resources->m_SurfaceCache->GetCardDataBuffer()->GetDescId(), primaryViewCBV,
                m_resources->m_SceneAggregator->GetGatheredBufferId());
        }
        // Pass Object Grid Debug - Vis
        if (false)
        {
            auto maxWorldBound  = m_globalDF->GetWorldBoundMax(0);
            auto minWorldBound  = m_globalDF->GetWorldBoundMin(0);
            auto voxelsPerWidth = m_globalDF->GetVoxelsPerSide(0);
            m_resources->m_Debugger->RenderValidObjectGrids(builder, &resDebugObjGridVis, &resGlobalObjectGrid,
                maxWorldBound, minWorldBound, voxelsPerWidth, primaryViewCBV);
        }

        // Pass Object Grid Debug
        if (false)
        {
            auto& resDirectLightingAtlas = m_resources->m_SurfaceCache->GetRDGDirectLightingAtlas();
            auto& resAlbedoAtlas         = m_resources->m_SurfaceCache->GetRDGAlbedoAtlas();
            auto& resDepthAtlas          = m_resources->m_SurfaceCache->GetRDGDepthAtlas();
            auto  maxWorldBound          = m_globalDF->GetWorldBoundMax(0);
            auto  minWorldBound          = m_globalDF->GetWorldBoundMin(0);
            auto  mdfDataId              = m_resources->m_SceneAggregator->GetGatheredBufferId();
            auto  cardDataId             = m_resources->m_SurfaceCache->GetCardDataBuffer()->GetDescId();

            auto  gdfResolution       = m_globalDF->GetClipmapWidth(0);
            auto  voxelsPerWidth      = m_globalDF->GetVoxelsPerSide(0);
            auto  cardResolution      = m_resources->m_SurfaceCache->GetCardResolution();
            auto  cardAtlasResolution = m_resources->m_SurfaceCache->GetCardAtlasResolution();

            m_resources->m_Debugger->RenderSceneFromSamplingObjectGrids(builder, &resDebugObjGridOut,
                &resDirectLightingAtlas, &resAlbedoAtlas, &resDepthAtlas, &resGlobalDFGen, &resGlobalObjectGrid,
                primaryViewCBV, cardDataId, mdfDataId, Vector2u(rtWidth, rtHeight), maxWorldBound, minWorldBound,
                cardResolution, cardAtlasResolution, voxelsPerWidth, gdfResolution);
        }

        // Pass Debug
        {
            auto& resDirectRadiance = m_resources->m_SurfaceCache->GetRDGShadowVisibilityAtlas();
            struct PushConst
            {
                u32 raymarchOutput = 0;
            } pc;
            AddFullScreenQuadPass<PushConst>(builder, "Ayanami.DebugPass", Internal::kIntShaderTableAyanami.CopyVS,
                Internal::kIntShaderTableAyanami.CopyFS, pc,
                [&](PushConst data, const FrameGraphPassContext& ctx) {
                    data.raymarchOutput = ctx.m_FgDesc->GetSRV(resRaymarchOutput);
                    SetRootSignature(data, ctx);
                })
                .AddRenderTarget(resRenderTargets)
                .AddReadResource(resDebugSCOut)
                .AddReadResource(resRaymarchOutput)
                .AddReadResource(resDirectRadiance)
                .AddReadResource(resDeferOut)
                .AddReadResource(resDebugObjGridOut)
                .AddReadResource(resDebugObjGridVis)
                .AddReadResource(resGNormal);
        }

        auto compiledFg = m_resources->m_FgCompiler.Compile(builder);
        m_resources->m_FgExecutor->ExecuteInSingleCmd(cmd, compiledFg);
        m_resources->m_Inited = true;
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
        m_resources->m_SceneAggregator->CollectScene(scene);
        m_resources->m_SurfaceCache->UpdateSceneCache(scene);
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