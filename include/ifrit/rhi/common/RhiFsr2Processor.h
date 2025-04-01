
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
#include "RhiForwardingTypes.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include <cstdint>

namespace Ifrit::Graphics::Rhi::FSR2
{

    struct RhiFSR2InitialzeArgs
    {
        u32 maxRenderWidth;
        u32 maxRenderHeight;
        u32 displayWidth;
        u32 displayHeight;
    };

    struct RhiFSR2DispatchArgs
    {
        Rhi::RhiTexture* color;
        Rhi::RhiTexture* depth;
        Rhi::RhiTexture* motion;
        Rhi::RhiTexture* exposure;
        Rhi::RhiTexture* reactiveMask;
        Rhi::RhiTexture* transparencyMask;
        Rhi::RhiTexture* output;
        float            deltaTime;
        float            jitterX;
        float            jitterY;
        float            camNear;
        float            camFar;
        float            camFovY;
        bool             reset;
    };

    class IFRIT_APIDECL RhiFsr2Processor
    {
    public:
        virtual ~RhiFsr2Processor()                                                                       = default;
        virtual void Init(const Rhi::FSR2::RhiFSR2InitialzeArgs& args)                                    = 0;
        virtual void GetJitters(float* jitterX, float* jitterY, u32 frameIdx, u32 rtWidth, u32 rtHeight)  = 0;
        virtual void Dispatch(const Rhi::RhiCommandList* cmd, const Rhi::FSR2::RhiFSR2DispatchArgs& args) = 0;
    };

} // namespace Ifrit::Graphics::Rhi::FSR2