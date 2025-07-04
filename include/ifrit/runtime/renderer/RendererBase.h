
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
#include "ifrit/runtime/common/Pch.h"

#include "ifrit/runtime/renderer/util/CascadeShadowMapPreproc.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/Scene.h"

#include "ifrit/runtime/scene/FrameCollector.h"

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
        using GPUTexture = RHI::RhiTextureRef;
        using SRVDesc    = RHI::RhiSRVDesc;
        std::mutex m_mutex;
        bool       m_initialized = false;
        GPUTexture m_blueNoise;
        SRVDesc    m_blueNoiseSRV = 0;
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
    enum class OverrideMaterialCulling
    {
        None,
        ForcedCullFront,
        ForcedCullBack,
        ForcedCullNone
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

        AntiAliasingType          m_AntiAliasingType        = AntiAliasingType::None;
        IndirectLightingType      m_IndirectLightingType    = IndirectLightingType::HBAO;
        RendererVisualizationType m_VisualizationType       = RendererVisualizationType::Default;
        OverrideMaterialCulling   m_OverrideMaterialCulling = OverrideMaterialCulling::None;
        ShadowConfig              m_ShadowConfig;
        f32                       m_SuperSamplingRate = 1.0f;
    };

    // TODO: move render graph to here
    class IFRIT_APIDECL RendererBase
    {
        using RenderTargets        = RHI::RhiRenderTargets;
        using GPUCommandSubmission = RHI::RhiTaskSubmission;

    protected:
        IApplication*              m_app;
        const RendererConfig*      m_config = nullptr;
        ImmutableRendererResources m_immRes;

    protected:
        RendererBase(IApplication* app) : m_app(app) {}

        inline void GetSupersampledRenderArea(
            const RenderTargets* finalRenderTargets, u32* renderWidth, u32* renderHeight)
        {
            *renderWidth = static_cast<u32>(finalRenderTargets->GetRenderArea().width / m_config->m_SuperSamplingRate);
            *renderHeight =
                static_cast<u32>(finalRenderTargets->GetRenderArea().height / m_config->m_SuperSamplingRate);
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
        virtual Owner<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait) = 0;

        virtual void                        EndFrame(const Vec<GPUCommandSubmission*>& cmdToWait);
        virtual Owner<GPUCommandSubmission> BeginFrame();
    };
} // namespace Ifrit::Runtime