
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
    inline Vector3f ConcentricOctahedralTransform(const Vector2f& sample)
    {
        // https://zhuanlan.zhihu.com/p/408898601
        // https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf

        constexpr f32 PI = 3.14159265358979323846f;

        Vector2f      sampleOffset = sample * 2.0f - Vector2f(1.0f);
        if (sampleOffset.x == 0.0f && sampleOffset.y == 0.0f)
        {
            return Vector3f(0.0f);
        }

        f32 u = sampleOffset.x;
        f32 v = sampleOffset.y;
        f32 d = 1.0f - std::abs(u) - std::abs(v);
        f32 r = 1.0f - std::abs(d);

        f32 z     = ((d > 0.0f) ? 1.0f : -1.0f) * (1.0f - r * r);
        f32 theta = PI / 4.0f * ((abs(v) - abs(u)) / (r + 1.0f));
        f32 sinT  = std::sin(theta) * ((v >= 0.0f) ? 1.0f : -1.0f);
        f32 cosT  = std::cos(theta) * ((u >= 0.0f) ? 1.0f : -1.0f);
        f32 x     = cosT * r * std::sqrt(2.0f - z * z);
        f32 y     = sinT * r * std::sqrt(2.0f - z * z);
        return Vector3f(x, y, z);
    }
} // namespace Ifrit::Math