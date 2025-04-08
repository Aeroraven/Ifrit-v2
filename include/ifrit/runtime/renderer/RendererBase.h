
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

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/runtime/renderer/util/CascadeShadowMapPreproc.h"

#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/Scene.h"

#include "ifrit/runtime/scene/FrameCollector.h"
#include "ifrit/rhi/common/RhiLayer.h"

#include <mutex>

namespace Ifrit::Runtime
{

    struct SceneCollectConfig
    {
        f32 projectionTranslateX = 0.0f;
        f32 projectionTranslateY = 0.0f;
    };

    struct ImmutableRendererResources
    {
        using GPUTexture = Graphics::Rhi::RhiTextureRef;
        using GPUBindId  = Graphics::Rhi::RhiDescHandleLegacy;
        std::mutex                 m_mutex;
        bool                       m_initialized = false;
        GPUTexture                 m_blueNoise;
        std::shared_ptr<GPUBindId> m_blueNoiseSRV = nullptr;
    };

    enum class AntiAliasingType
    {
        None,
        TAA,
        FSR2
    };
    enum class RendererVisualizationType
    {
        Default,
        Triangle,
        SwHwMaps
    };
    enum class IndirectLightingType
    {
        HBAO,
        SSGI
    };

    struct RendererConfig
    {
        struct ShadowConfig
        {
            IF_CONSTEXPR static u32 k_maxShadowMaps = 256;
            f32                     m_maxDistance   = 5.0f;
            u32                     m_csmCount      = 4;
            Array<f32, 4>           m_csmSplits     = { 0.067f, 0.133f, 0.267f, 0.533f };
            Array<f32, 4>           m_csmBorders    = { 0.08f, 0.05f, 0.0f, 0.0f };
        };

        AntiAliasingType          m_antiAliasingType     = AntiAliasingType::None;
        IndirectLightingType      m_indirectLightingType = IndirectLightingType::HBAO;
        RendererVisualizationType m_visualizationType    = RendererVisualizationType::Default;
        ShadowConfig              m_shadowConfig;
        f32                       m_superSamplingRate = 1.0f;
    };

    // TODO: move render graph to here
    class IFRIT_APIDECL RendererBase
    {
        using RenderTargets        = Graphics::Rhi::RhiRenderTargets;
        using GPUCommandSubmission = Graphics::Rhi::RhiTaskSubmission;

    protected:
        IApplication*              m_app;
        const RendererConfig*      m_config = nullptr;
        ImmutableRendererResources m_immRes;

    protected:
        RendererBase(IApplication* app) : m_app(app) {}

        inline void GetSupersampledRenderArea(
            const RenderTargets* finalRenderTargets, u32* renderWidth, u32* renderHeight)
        {
            *renderWidth = static_cast<u32>(finalRenderTargets->GetRenderArea().width / m_config->m_superSamplingRate);
            *renderHeight =
                static_cast<u32>(finalRenderTargets->GetRenderArea().height / m_config->m_superSamplingRate);
        }

        virtual void PrepareImmutableResources();

        virtual void BuildPipelines(
            PerFrameData& perframeData, GraphicsShaderPassType passType, RenderTargets* renderTargets);
        virtual void PrepareDeviceResources(PerFrameData& perframeData, RenderTargets* renderTargets);
        virtual void UpdateLastFrameTransforms(PerFrameData& perframeData);
        virtual void RecreateGBuffers(PerFrameData& perframeData, RenderTargets* renderTargets);

        virtual void CollectPerframeData(PerFrameData& perframeData, Scene* scene, Camera* camera,
            GraphicsShaderPassType passType, RenderTargets* renderTargets, const SceneCollectConfig& config);

        inline void  SetRendererConfig(const RendererConfig* config) { m_config = config; }

    public:
        virtual Uref<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait) = 0;

        virtual void                       EndFrame(const Vec<GPUCommandSubmission*>& cmdToWait);
        virtual Uref<GPUCommandSubmission> BeginFrame();
    };
} // namespace Ifrit::Runtime