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
#include "ifrit/runtime/common/Pch.h"

#include "ifrit/core/file/FileOps.h"
#include "ifrit/runtime/renderer/RendererUtil.h"
#include "ifrit/runtime/renderer/SyaroRenderer.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"

namespace Ifrit::Runtime
{

    struct BaseForwardRendererResources;

    // BaseForwardRenderer is a renderer that implements the forward rendering technique.
    // It's designed to add the minimal set of features required for rendering a scene without
    // advanced device support. For advanced devices, please use SyaroV1.
    class IFRIT_APIDECL BaseForwardRenderer : public RendererBase
    {

        using RenderTargets        = RHI::RhiRenderTargets;
        using GPUCommandSubmission = RHI::RhiTaskSubmission;
        using GPUCmdBuffer         = RHI::RhiCommandList;

    private:
        BaseForwardRendererResources* m_Resources = nullptr;

    private:
        void InitRenderer();
        void SetupAndRunFrameGraph(
            Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);

    public:
        BaseForwardRenderer(IApplication* app);
        virtual ~BaseForwardRenderer();

        virtual Owner<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait) override;
    };

} // namespace Ifrit::Runtime
