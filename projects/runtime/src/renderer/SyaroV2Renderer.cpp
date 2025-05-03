
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

#include "ifrit/runtime/renderer/SyaroV2Renderer.h"
#include "ifrit/runtime/renderer/RendererUtil.h"
using namespace Ifrit::Graphics::Rhi;

namespace Ifrit::Runtime
{

    f32 GetFrameTimestampMili()
    {
        auto frameTimestampRaw = std::chrono::high_resolution_clock::now();
        auto frameTimestampMicro =
            std::chrono::duration_cast<std::chrono::microseconds>(frameTimestampRaw.time_since_epoch());
        auto  frameTimestampMicroCount = frameTimestampMicro.count();
        float frameTimestampMili       = frameTimestampMicroCount / 1000.0f;
        return frameTimestampMili;
    }

    void PrepareTemporalAAJitter(const RendererConfig& renderCfg, SceneCollectConfig& sceneConfig,
        PerFrameData& perFrame, u32 frameId, u32 actualRw, u32 actualRh, u32 outputRw, FSR2::RhiFsr2Processor* fsr2proc)
    {
        if (renderCfg.m_AntiAliasingType == AntiAliasingType::TAA)
        {
            auto haltonX = RendererConsts::cHalton2[frameId % RendererConsts::cHalton2.size()];
            auto haltonY = RendererConsts::cHalton3[frameId % RendererConsts::cHalton3.size()];

            sceneConfig.projectionTranslateX = (haltonX * 2.0f - 1.0f) / actualRw;
            sceneConfig.projectionTranslateY = (haltonY * 2.0f - 1.0f) / actualRh;

            perFrame.m_taaJitterX = sceneConfig.projectionTranslateX * 0.5f;
            perFrame.m_taaJitterY = sceneConfig.projectionTranslateY * 0.5f;
        }
        else if (renderCfg.m_AntiAliasingType == AntiAliasingType::FSR2)
        {
            f32 jx, jy;
            fsr2proc->GetJitters(&jx, &jy, frameId, actualRw, outputRw);
            sceneConfig.projectionTranslateX = 2.0f * jx / actualRw;
            sceneConfig.projectionTranslateY = 2.0f * jy / actualRh;
            perFrame.m_taaJitterX            = jx;
            perFrame.m_taaJitterY            = jy;
        }
        else
        {
            sceneConfig.projectionTranslateX = 0.0f;
            sceneConfig.projectionTranslateY = 0.0f;
            perFrame.m_taaJitterX            = 0.0f;
            perFrame.m_taaJitterY            = 0.0f;
        }

        // If in debug view, we ignore all jitters
        if (renderCfg.m_VisualizationType != RendererVisualizationType::Default)
        {
            sceneConfig.projectionTranslateX = 0.0f;
            sceneConfig.projectionTranslateY = 0.0f;
            perFrame.m_taaJitterX            = 0.0f;
            perFrame.m_taaJitterY            = 0.0f;
        }
    }

    PerFrameData::PerViewData* GetPrimaryView(PerFrameData& perframeData)
    {
        for (auto& view : perframeData.m_views)
        {
            if (view.m_viewType == PerFrameData::ViewType::Primary)
            {
                return &view;
            }
        }
        iError("No primary view found in PerFrameData.");
        std::abort();
        return &perframeData.m_views[0];
    }

    struct SyaroV2RendererResources
    {
        SyaroRenderRole              m_RenderRole = SyaroRenderRole::FullProcess;
        RendererConfig               m_RendererConfig;

        PerFrameData*                m_ActivePerFrameData    = nullptr;
        PerFrameData::PerViewData*   m_ActivePrimaryViewData = nullptr;

        Uref<FSR2::RhiFsr2Processor> m_FSR2Proc = nullptr;
    };

    IFRIT_APIDECL SyaroV2Renderer::SyaroV2Renderer(IApplication* app) : RendererBase(app)
    {
        m_Res = new SyaroV2RendererResources();

        m_Res->m_FSR2Proc = m_app->GetRhi()->CreateFsr2Processor();
    }

    IFRIT_APIDECL      SyaroV2Renderer::~SyaroV2Renderer() { delete m_Res; }

    IFRIT_APIDECL void SyaroV2Renderer::SetRenderRole(u32 role)
    {
        m_Res->m_RenderRole = static_cast<SyaroRenderRole>(role);
    }

    Uref<RhiTaskSubmission> SyaroV2Renderer::Render(Scene* scene, Camera* camera, RhiRenderTargets* renderTargets,
        const RendererConfig& config, const Vec<RhiTaskSubmission*>& cmdToWait)
    {
        PrepareImmutableResources();

        m_Res->m_RendererConfig     = config;
        m_config                    = &m_Res->m_RendererConfig;
        m_Res->m_ActivePerFrameData = scene->GetPerFrameData().get();

        auto& perFrameData                         = *m_Res->m_ActivePerFrameData;
        auto  frameId                              = perFrameData.m_frameId;
        perFrameData.m_frameTimestamp[frameId % 2] = GetFrameTimestampMili();

        u32 actualRw = 0, actualRh = 0;
        GetSupersampledRenderArea(renderTargets, &actualRw, &actualRh);

        SceneCollectConfig sceneConfig;
        auto               outputWidth = renderTargets->GetRenderArea().width;
        PrepareTemporalAAJitter(
            config, sceneConfig, perFrameData, frameId, actualRw, actualRh, outputWidth, m_Res->m_FSR2Proc.get());

        // Start render here
        auto rhi       = m_app->GetRhi();
        auto drawQueue = rhi->GetQueue(RhiQueueCapability::RhiQueue_Graphics);

        auto drawTask = drawQueue->RunAsyncCommand(
            [&](const RhiCommandList* cmd) { SetupAndRunFrameGraph(scene, perFrameData, renderTargets, cmd); },
            cmdToWait, {});

        // Proceed to next frame, on cpu side
        if (m_Res->m_RenderRole & SyaroRenderRole::FullProcess)
        {
            perFrameData.m_frameId++;
        }

        return drawTask;
    }

    IFRIT_APIDECL void SyaroV2Renderer::SetupAndRunFrameGraph(
        Scene* scene, PerFrameData& perframe, RhiRenderTargets* renderTargets, const RhiCommandList* cmd)
    {
    }

} // namespace Ifrit::Runtime