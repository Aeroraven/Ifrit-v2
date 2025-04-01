
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
#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/scene/FrameCollector.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL AmbientOcclusionPass
    {
        using ComputePass   = Graphics::Rhi::RhiComputePass;
        using CommandBuffer = Graphics::Rhi::RhiCommandList;
        using GPUBindId     = Graphics::Rhi::RhiDescHandleLegacy;
        using GPUShader     = Graphics::Rhi::RhiShader;

    private:
        IApplication* m_app;
        ComputePass*  m_hbaoPass = nullptr;
        ComputePass*  m_ssgiPass = nullptr;

        void          SetupHBAOPass();
        void          SetupSSGIPass();
        GPUShader*    CreateShaderFromFile(
               const String& shaderPath, const String& entry, Graphics::Rhi::RhiShaderStage stage);

    public:
        AmbientOcclusionPass(IApplication* app) : m_app(app) {}
        void RenderHBAO(const CommandBuffer* cmd, u32 width, u32 height, GPUBindId* depthSamp, GPUBindId* normalSamp,
            u32 aoTex, GPUBindId* perframeData);

        void RenderSSGI(const CommandBuffer* cmd, u32 width, u32 height, GPUBindId* perframeData, u32 depthHizMinUAV,
            u32 depthHizMaxUAV, GPUBindId* normalSRV, u32 aoUAV, GPUBindId* albedoSRV, u32 hizTexW, u32 hizTexH,
            u32 numLods, GPUBindId* blueNoiseSRV);
    };
} // namespace Ifrit::Runtime