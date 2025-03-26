
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/rhi/common/RhiFsr2Processor.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
namespace Ifrit::Graphics::VulkanGraphics::FSR2
{
    struct FSR2Context;

    class IFRIT_APIDECL FSR2Processor : public Rhi::FSR2::RhiFsr2Processor
    {
    private:
        FSR2Context*                    m_context       = nullptr;
        EngineContext*                  m_engineContext = nullptr;
        Rhi::FSR2::RhiFSR2InitialzeArgs m_args;

    public:
        FSR2Processor(EngineContext* ctx);
        ~FSR2Processor();
        void Init(const Rhi::FSR2::RhiFSR2InitialzeArgs& args) override;
        void Dispatch(const Rhi::RhiCommandList* cmd, const Rhi::FSR2::RhiFSR2DispatchArgs& args) override;
        void GetJitters(float* jitterX, float* jitterY, u32 frameIdx, u32 rtWidth, u32 dispWidth) override;
    };
} // namespace Ifrit::Graphics::VulkanGraphics::FSR2