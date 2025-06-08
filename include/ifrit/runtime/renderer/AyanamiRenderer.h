
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


#include "ifrit/core/file/FileOps.h"
#include "ifrit/runtime/renderer/RendererUtil.h"
#include "ifrit/runtime/renderer/SyaroRenderer.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"
#include <algorithm>
#include <bit>

#include "ayanami/AyanamiGlobalDF.h"

using Ifrit::SizeCast;
using Ifrit::Math::DivRoundUp;

namespace Ifrit::Runtime
{

    struct AyanamiRendererResources;
    class IFRIT_APIDECL AyanamiRenderer : public RendererBase
    {

        using RenderTargets        = Graphics::Rhi::RhiRenderTargets;
        using GPUCommandSubmission = Graphics::Rhi::RhiTaskSubmission;
        using GPUBuffer            = Graphics::Rhi::RhiBuffer;
        using GPUBindId            = Graphics::Rhi::RhiDescHandleLegacy;
        using GPUDescRef           = Graphics::Rhi::RhiBindlessDescriptorRef;
        using ComputePass          = Graphics::Rhi::RhiComputePass;
        using DrawPass             = Graphics::Rhi::RhiGraphicsPass;
        using GPUShader            = Graphics::Rhi::RhiShader;
        using GPUTexture           = Graphics::Rhi::RhiTexture;
        using GPUCmdBuffer         = Graphics::Rhi::RhiCommandList;

    private:
        Uref<SyaroRenderer>            m_VGRenderer;
        AyanamiRendererResources*      m_Resources = nullptr;

        Ayanami::AyanamiRenderConfig   m_SelfRenderConfig;
        Uref<Ayanami::AyanamiGlobalDF> m_GlobalDF = nullptr;

    private:
        void InitRenderer();
        void PrepareResources(RenderTargets* renderTargets, const RendererConfig& config);
        void SetupAndRunFrameGraph(
            Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);

    public:
        AyanamiRenderer(IApplication* app, Ayanami::AyanamiRenderConfig config)
            : RendererBase(app), m_VGRenderer(std::make_unique<SyaroRenderer>(app)), m_SelfRenderConfig(config)
        {
            m_VGRenderer->SetRenderRole(SyaroRenderRole::GBuffer | SyaroRenderRole::Shadowing);
            InitRenderer();
            m_GlobalDF = std::make_unique<Ayanami::AyanamiGlobalDF>(config, app);
        }
        virtual ~AyanamiRenderer();

        virtual Uref<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait) override;
    };

} // namespace Ifrit::Runtime