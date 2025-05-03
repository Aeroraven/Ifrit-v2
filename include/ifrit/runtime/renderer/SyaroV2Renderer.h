
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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
#include "AmbientOcclusionPass.h"
#include "PbrAtmosphereRenderer.h"
#include "RendererBase.h"
#include "framegraph/FrameGraph.h"
#include "ifrit/core/base/IfritBase.h"

#include "ifrit/runtime/renderer/syaro/SyaroEnums.h"

namespace Ifrit::Runtime
{
    // The refactored version for SyaroRenderer
    // The V2 version is scheduled with following features:
    // - RDG Driven

    struct SyaroV2RendererResources;
    class IFRIT_RUNTIME_API SyaroV2Renderer : public RendererBase, public NonCopyable
    {
    private:
        SyaroV2RendererResources* m_Res = nullptr;

    private:
        void SetupAndRunFrameGraph(Scene* scene, PerFrameData& perframe, Graphics::Rhi::RhiRenderTargets* renderTargets,
            const Graphics::Rhi::RhiCommandList* cmd);

    public:
        SyaroV2Renderer(IApplication* app);
        virtual ~SyaroV2Renderer();

        void                                           SetRenderRole(u32 role);
        virtual Uref<Graphics::Rhi::RhiTaskSubmission> Render(Scene* scene, Camera* camera,
            Graphics::Rhi::RhiRenderTargets* renderTargets, const RendererConfig& config,
            const Vec<Graphics::Rhi::RhiTaskSubmission*>& cmdToWait) override;
    };
} // namespace Ifrit::Runtime