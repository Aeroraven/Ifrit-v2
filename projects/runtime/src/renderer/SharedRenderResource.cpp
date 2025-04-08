
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
#include "ifrit/runtime/renderer/SharedRenderResource.h"
#include "ifrit.shader/SamplerUtils.SharedConst.h"
#include "ifrit/core/typing/Util.h"

#include "ifrit/runtime/base/Base.h"
using namespace Ifrit::Graphics::Rhi;

namespace Ifrit::Runtime
{
    struct SharedRenderResourceData
    {
        RhiSamplerRef              m_LinearClampSampler   = nullptr;
        RhiSamplerRef              m_NearestClampSampler  = nullptr;
        RhiSamplerRef              m_LinearRepeatSampler  = nullptr;
        RhiSamplerRef              m_NearestRepeatSampler = nullptr;

        Graphics::Rhi::RhiBackend* m_RhiBackend = nullptr;
    };

    IFRIT_APIDECL SharedRenderResource::SharedRenderResource(Graphics::Rhi::RhiBackend* rhi)
    {
        m_Data                        = new SharedRenderResourceData();
        m_Data->m_RhiBackend          = rhi;
        m_Data->m_LinearClampSampler  = rhi->CreateSampler(RhiSamplerFilter::Linear, RhiSamplerWrapMode::Clamp, true);
        m_Data->m_NearestClampSampler = rhi->CreateSampler(RhiSamplerFilter::Nearest, RhiSamplerWrapMode::Clamp, true);
        m_Data->m_LinearRepeatSampler = rhi->CreateSampler(RhiSamplerFilter::Linear, RhiSamplerWrapMode::Repeat, true);
        m_Data->m_NearestRepeatSampler =
            rhi->CreateSampler(RhiSamplerFilter::Nearest, RhiSamplerWrapMode::Repeat, true);

        iAssertion(m_Data->m_LinearClampSampler->GetDescId() == SamplerUtils::sLinearClamp,
            "Sampler ID for `sLinearClamp` mismatches!");
        iAssertion(m_Data->m_NearestClampSampler->GetDescId() == SamplerUtils::sNearestClamp,
            "Sampler ID for `sNearestClamp` mismatches!");
        iAssertion(m_Data->m_LinearRepeatSampler->GetDescId() == SamplerUtils::sLinearRepeat,
            "Sampler ID for `sLinearRepeat` mismatches!");
        iAssertion(m_Data->m_NearestRepeatSampler->GetDescId() == SamplerUtils::sNearestRepeat,
            "Sampler ID for `sNearestRepeat` mismatches!");
    }

    IFRIT_APIDECL SharedRenderResource::~SharedRenderResource()
    {
        delete m_Data;
        m_Data = nullptr;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiSamplerDesc SharedRenderResource::GetLinearClampSamplerDesc()
    {
        return m_Data->m_LinearClampSampler->GetDescId();
    }

    IFRIT_APIDECL Graphics::Rhi::RhiSamplerDesc SharedRenderResource::GetNearestClampSamplerDesc()
    {
        return m_Data->m_NearestClampSampler->GetDescId();
    }

    IFRIT_APIDECL Graphics::Rhi::RhiSamplerDesc SharedRenderResource::GetLinearRepeatSamplerDesc()
    {
        return m_Data->m_LinearRepeatSampler->GetDescId();
    }

    IFRIT_APIDECL Graphics::Rhi::RhiSamplerDesc SharedRenderResource::GetNearestRepeatSamplerDesc()
    {
        return m_Data->m_NearestRepeatSampler->GetDescId();
    }

    // Only for refactoring,
    IFRIT_APIDECL Graphics::Rhi::RhiSamplerRef SharedRenderResource::GetLinearClampSampler()
    {
        return m_Data->m_LinearClampSampler;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiSamplerRef SharedRenderResource::GetNearestClampSampler()
    {
        return m_Data->m_NearestClampSampler;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiSamplerRef SharedRenderResource::GetLinearRepeatSampler()
    {
        return m_Data->m_LinearRepeatSampler;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiSamplerRef SharedRenderResource::GetNearestRepeatSampler()
    {
        return m_Data->m_NearestRepeatSampler;
    }

} // namespace Ifrit::Runtime