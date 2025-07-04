
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
    class IFRIT_APIDECL AmbientOcclusionPass
    {
        using ComputePass   = RHI::RhiComputePass;
        using CommandBuffer = RHI::RhiCommandList;
        using SRVDesc       = RHI::RhiSRVDesc;
        using CBVDesc       = RHI::RhiCBVDesc;
        using GPUShader     = RHI::RhiShader;

    private:
        IApplication* m_app;
        ComputePass*  m_hbaoPass = nullptr;
        ComputePass*  m_ssgiPass = nullptr;

        void          SetupHBAOPass();
        void          SetupSSGIPass();
        GPUShader*    GetInternalShader(const char* name);

    public:
        AmbientOcclusionPass(IApplication* app) : m_app(app) {}
        void RenderHBAO(const CommandBuffer* cmd, u32 width, u32 height, SRVDesc depthSamp, SRVDesc normalSamp,
            u32 aoTex, CBVDesc perframeData);

        void RenderSSGI(const CommandBuffer* cmd, u32 width, u32 height, CBVDesc perframeData, u32 depthHizMinUAV,
            u32 depthHizMaxUAV, SRVDesc normalSRV, u32 aoUAV, u32 finalLightingSRV, u32 hizTexW, u32 hizTexH,
            u32 numLods, SRVDesc blueNoiseSRV, SRVDesc albedoSRV);
    };
} // namespace Ifrit::Runtime