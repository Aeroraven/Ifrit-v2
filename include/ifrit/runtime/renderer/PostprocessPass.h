
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
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/scene/FrameCollector.h"

namespace Ifrit::Runtime
{
    struct PostprocessPassConfig
    {
        String fragPath;
        u32    numPushConstants;
        u32    numDescriptorSets;
        bool   isComputeShader = false;
    };

    class IFRIT_APIDECL PostprocessPass
    {
    protected:
        using DrawPass       = RHI::RhiGraphicsPass;
        using ComputePass    = RHI::RhiComputePass;
        using RenderTargets  = RHI::RhiRenderTargets;
        using GPUShader      = RHI::RhiShader;
        using GPUCmdBuffer   = RHI::RhiCommandList;
        using GPUBindlessRef = RHI::RhiBindlessDescriptorRef;

    protected:
        using PipeConfig = PipelineAttachmentConfigs;
        using PipeHash   = PipelineAttachmentConfigsHash;
        PostprocessPassConfig                          m_cfg;
        IApplication*                                  m_app;
        CustomHashMap<PipeConfig, DrawPass*, PipeHash> m_renderPipelines;
        ComputePass*                                   m_computePipeline = nullptr;

    protected:
        GPUShader*   CreateInternalShader(const char* name);
        DrawPass*    SetupRenderPipeline(RenderTargets* renderTargets);
        ComputePass* SetupComputePipeline();
        void         RenderInternal(PerFrameData* perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd,
                    const void* pushConstants, const Vec<GPUBindlessRef*>& bindDescs, const String& scopeName);

    public:
        PostprocessPass(IApplication* app, const PostprocessPassConfig& cfg);
        virtual ~PostprocessPass() = default;
    };
} // namespace Ifrit::Runtime
