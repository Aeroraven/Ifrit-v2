
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
#include "ifrit/runtime/base/Base.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/rhi/common/RhiLayer.h"
namespace Ifrit::Runtime
{
    struct SharedRenderResourceData;
    class IFRIT_RUNTIME_API SharedRenderResource : public NonCopyable
    {
    private:
        SharedRenderResourceData* m_Data = nullptr;

    public:
        SharedRenderResource(Graphics::Rhi::RhiBackend* rhi);
        virtual ~SharedRenderResource();

        Graphics::Rhi::RhiSamplerDesc GetLinearClampSamplerDesc();
        Graphics::Rhi::RhiSamplerDesc GetNearestClampSamplerDesc();
        Graphics::Rhi::RhiSamplerDesc GetLinearRepeatSamplerDesc();
        Graphics::Rhi::RhiSamplerDesc GetNearestRepeatSamplerDesc();

        // Only for refactoring,
        Graphics::Rhi::RhiSamplerRef  GetLinearClampSampler();
        Graphics::Rhi::RhiSamplerRef  GetNearestClampSampler();
        Graphics::Rhi::RhiSamplerRef  GetLinearRepeatSampler();
        Graphics::Rhi::RhiSamplerRef  GetNearestRepeatSampler();
    };
} // namespace Ifrit::Runtime