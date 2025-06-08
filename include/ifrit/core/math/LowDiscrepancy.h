
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
#include "../platform/ApiConv.h"
#include "VectorOps.h"
#include "ifrit/core/base/IfritBase.h"

namespace Ifrit::Math
{
    IF_FORCEINLINE f32 VanDeCorputRadicalInverse2(u32 bits)
    {
        // Reference:
        // https://github.com/Nadrin/PBR/blob/master/data/shaders/hlsl/spmap.hlsl
        // https://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html

        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
    }

    IF_FORCEINLINE Vector2f Hammersley2d(u32 x, u32 N)
    {
        f32 u1 = f32(x) / f32(N);
        f32 u2 = VanDeCorputRadicalInverse2(x);
        return Vector2f(u1, u2);
    }
} // namespace Ifrit::Math