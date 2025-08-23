
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

#include "RhiBaseTypes.h"
#include "RhiResource.h"

namespace Ifrit::RHI
{
    struct RhiUAVBarrier
    {
        ERhiResourceType m_type;
        union
        {
            RhiBuffer*  m_buffer;
            RhiTexture* m_texture;
        };
    };

    struct RhiTransitionBarrier
    {
        ERhiResourceType m_type;
        union
        {
            RhiBuffer*  m_buffer = nullptr;
            RhiTexture* m_texture;
        };
        RhiImageSubResource m_subResource = { 0, 0, 1, 1 };
        ERhiResourceState   m_srcState    = ERhiResourceState::Undefined;
        ERhiResourceState   m_dstState    = ERhiResourceState::Undefined;

        RhiTransitionBarrier() { m_texture = nullptr; }
    };

    struct RhiResourceBarrier
    {
        ERhiBarrierType m_type = ERhiBarrierType::UAVAccess;
        union
        {
            RhiUAVBarrier        m_uav;
            RhiTransitionBarrier m_transition;
        };
        RhiResourceBarrier() { m_uav = {}; }
    };
} // namespace Ifrit::RHI