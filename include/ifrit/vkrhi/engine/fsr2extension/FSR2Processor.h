
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
#include "ifrit/core/logging/Logging.h"
#include "ifrit/rhi/common/RhiFsr2Processor.h"
#include "ifrit/vkrhi/engine/vkrenderer/EngineContext.h"
namespace Ifrit::RHI::VulkanAdapter::FSR2
{
    struct FSR2Context;

    class IFRIT_APIDECL FSR2Processor : public RHI::FSR2::RhiFsr2Processor
    {
    private:
        FSR2Context*                    m_context       = nullptr;
        EngineContext*                  m_engineContext = nullptr;
        RHI::FSR2::RhiFSR2InitialzeArgs m_args;

    public:
        FSR2Processor(EngineContext* ctx);
        ~FSR2Processor();
        void Init(const RHI::FSR2::RhiFSR2InitialzeArgs& args) override;
        void Dispatch(const RHI::RhiCommandListContext* cmd, const RHI::FSR2::RhiFSR2DispatchArgs& args) override;
        void GetJitters(float* jitterX, float* jitterY, u32 frameIdx, u32 rtWidth, u32 dispWidth) override;
    };
} // namespace Ifrit::RHI::VulkanAdapter::FSR2