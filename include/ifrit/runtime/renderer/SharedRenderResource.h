
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
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime
{
    struct SharedRenderResourceData;
    class IFRIT_RUNTIME_API SharedRenderResource : public NonCopyable
    {
    private:
        SharedRenderResourceData* m_Data = nullptr;

    public:
        SharedRenderResource(RHI::RhiBackend* rhi);
        virtual ~SharedRenderResource();

        RHI::RhiSamplerDesc GetLinearClampSamplerDesc();
        RHI::RhiSamplerDesc GetNearestClampSamplerDesc();
        RHI::RhiSamplerDesc GetLinearRepeatSamplerDesc();
        RHI::RhiSamplerDesc GetNearestRepeatSamplerDesc();

        // Only for refactoring,
        RHI::RhiSamplerRef  GetLinearClampSampler();
        RHI::RhiSamplerRef  GetNearestClampSampler();
        RHI::RhiSamplerRef  GetLinearRepeatSampler();
        RHI::RhiSamplerRef  GetNearestRepeatSampler();
    };
} // namespace Ifrit::Runtime