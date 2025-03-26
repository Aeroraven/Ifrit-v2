
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
#include "../util/ApiConv.h"
#include "VectorDefs.h"
#include <algorithm>
#include <cmath>

namespace Ifrit::Math
{
// Element wise ops
#define ELEMENTWISE_VECTOR_OP(op)                                              \
    template <typename T>                                                      \
    inline CoreVec2<T> operator op(const CoreVec2<T>& a, const CoreVec2<T>& b) \
    {                                                                          \
        return CoreVec2<T>{ a.x op b.x, a.y op b.y };                          \
    }                                                                          \
    template <typename T>                                                      \
    inline CoreVec3<T> operator op(const CoreVec3<T>& a, const CoreVec3<T>& b) \
    {                                                                          \
        return CoreVec3<T>{ a.x op b.x, a.y op b.y, a.z op b.z };              \
    }                                                                          \
    template <typename T>                                                      \
    inline CoreVec4<T> operator op(const CoreVec4<T>& a, const CoreVec4<T>& b) \
    {                                                                          \
        return CoreVec4<T>{ a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w };  \
    }
    ELEMENTWISE_VECTOR_OP(+);
    ELEMENTWISE_VECTOR_OP(-);
    ELEMENTWISE_VECTOR_OP(*);
    ELEMENTWISE_VECTOR_OP(/);
#undef ELEMENTWISE_VECTOR_OP

// Scalar ops
#define SCALAR_VECTOR_OP(op)                                          \
    template <typename T>                                             \
    inline CoreVec2<T> operator op(const CoreVec2<T>& a, T b)         \
    {                                                                 \
        return CoreVec2<T>{ a.x op b, a.y op b };                     \
    }                                                                 \
    template <typename T>                                             \
    inline CoreVec3<T> operator op(const CoreVec3<T>& a, T b)         \
    {                                                                 \
        return CoreVec3<T>{ a.x op b, a.y op b, a.z op b };           \
    }                                                                 \
    template <typename T>                                             \
    inline CoreVec4<T> operator op(const CoreVec4<T>& a, T b)         \
    {                                                                 \
        return CoreVec4<T>{ a.x op b, a.y op b, a.z op b, a.w op b }; \
    }
    SCALAR_VECTOR_OP(+);
    SCALAR_VECTOR_OP(-);
    SCALAR_VECTOR_OP(*);
    SCALAR_VECTOR_OP(/);
#undef SCALAR_VECTOR_OP

    // Element wise ops 2

#define ELEMENTWISE_VECTOR_OP2(op)                                        \
    template <typename T>                                                 \
    inline CoreVec2<T>& operator op(CoreVec2<T>& a, const CoreVec2<T>& b) \
    {                                                                     \
        a.x op b.x;                                                       \
        a.y op b.y;                                                       \
        return a;                                                         \
    }                                                                     \
    template <typename T>                                                 \
    inline CoreVec3<T>& operator op(CoreVec3<T>& a, const CoreVec3<T>& b) \
    {                                                                     \
        a.x op b.x;                                                       \
        a.y op b.y;                                                       \
        a.z op b.z;                                                       \
        return a;                                                         \
    }                                                                     \
    template <typename T>                                                 \
    inline CoreVec4<T>& operator op(CoreVec4<T>& a, const CoreVec4<T>& b) \
    {                                                                     \
        a.x op b.x;                                                       \
        a.y op b.y;                                                       \
        a.z op b.z;                                                       \
        a.w op b.w;                                                       \
        return a;                                                         \
    }
    ELEMENTWISE_VECTOR_OP2(+=);
    ELEMENTWISE_VECTOR_OP2(-=);
    ELEMENTWISE_VECTOR_OP2(*=);
    ELEMENTWISE_VECTOR_OP2(/=);

#undef ELEMENTWISE_VECTOR_OP2

// Scalar ops 2
#define SCALAR_VECTOR_OP2(op)                            \
    template <typename T>                                \
    inline CoreVec2<T>& operator op(CoreVec2<T>& a, T b) \
    {                                                    \
        a.x op b;                                        \
        a.y op b;                                        \
        return a;                                        \
    }                                                    \
    template <typename T>                                \
    inline CoreVec3<T>& operator op(CoreVec3<T>& a, T b) \
    {                                                    \
        a.x op b;                                        \
        a.y op b;                                        \
        a.z op b;                                        \
        return a;                                        \
    }                                                    \
    template <typename T>                                \
    inline CoreVec4<T>& operator op(CoreVec4<T>& a, T b) \
    {                                                    \
        a.x op b;                                        \
        a.y op b;                                        \
        a.z op b;                                        \
        a.w op b;                                        \
        return a;                                        \
    }

    SCALAR_VECTOR_OP2(+=);
    SCALAR_VECTOR_OP2(-=);
    SCALAR_VECTOR_OP2(*=);
    SCALAR_VECTOR_OP2(/=);

#undef SCALAR_VECTOR_OP2

    // Address ops
    template <typename T>
    inline T ElementAt(CoreVec2<T>& a, int i)
    {
        return i == 0 ? a.x : a.y;
    }
    template <typename T>
    inline T ElementAt(CoreVec3<T>& a, int i)
    {
        return i == 0 ? a.x : i == 1 ? a.y
                                     : a.z;
    }
    template <typename T>
    inline T ElementAt(CoreVec4<T>& a, int i)
    {
        return i == 0 ? a.x : i == 1 ? a.y
            : i == 2                 ? a.z
                                     : a.w;
    }

    // Normalize ops
    template <typename T>
    inline CoreVec2<T> Normalize(const CoreVec2<T>& a)
    {
        T len = sqrt(a.x * a.x + a.y * a.y);
        return CoreVec2<T>{ a.x / len, a.y / len };
    }
    template <typename T>
    inline CoreVec3<T> Normalize(const CoreVec3<T>& a)
    {
        T len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        return CoreVec3<T>{ a.x / len, a.y / len, a.z / len };
    }
    template <typename T>
    inline CoreVec4<T> Normalize(const CoreVec4<T>& a)
    {
        T len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
        return CoreVec4<T>{ a.x / len, a.y / len, a.z / len, a.w / len };
    }

    // Length ops
    template <typename T>
    inline T Length(const CoreVec2<T>& a)
    {
        return sqrt(a.x * a.x + a.y * a.y);
    }
    template <typename T>
    inline T Length(const CoreVec3<T>& a)
    {
        return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    }
    template <typename T>
    inline T Length(const CoreVec4<T>& a)
    {
        return sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }

    // Dot ops
    template <typename T>
    inline T Dot(const CoreVec2<T>& a, const CoreVec2<T>& b)
    {
        return a.x * b.x + a.y * b.y;
    }
    template <typename T>
    inline T Dot(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    template <typename T>
    inline T Dot(const CoreVec4<T>& a, const CoreVec4<T>& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // Lerp ops
    template <typename T>
    inline CoreVec2<T> Lerp(const CoreVec2<T>& a, const CoreVec2<T>& b, const T& t)
    {
        return a * (1 - t) + b * t;
    }
    template <typename T>
    inline CoreVec3<T> Lerp(const CoreVec3<T>& a, const CoreVec3<T>& b, const T& t)
    {
        return a * (1 - t) + b * t;
    }
    template <typename T>
    inline CoreVec4<T> Lerp(const CoreVec4<T>& a, const CoreVec4<T>& b, const T& t)
    {
        return a * (1 - t) + b * t;
    }

    // Clamp ops
    template <typename T>
    inline CoreVec2<T> Clamp(const CoreVec2<T>& a, const T& min, const T& max)
    {
        return CoreVec2<T>{ std::clamp(a.x, min, max), std::clamp(a.y, min, max) };
    }
    template <typename T>
    inline CoreVec3<T> Clamp(const CoreVec3<T>& a, const T& min, const T& max)
    {
        return CoreVec3<T>{ std::clamp(a.x, min, max), std::clamp(a.y, min, max), std::clamp(a.z, min, max) };
    }
    template <typename T>
    inline CoreVec4<T> Clamp(const CoreVec4<T>& a, const T& min, const T& max)
    {
        return CoreVec4<T>{ std::clamp(a.x, min, max), std::clamp(a.y, min, max), std::clamp(a.z, min, max),
            std::clamp(a.w, min, max) };
    }

    // Cross ops
    template <typename T>
    inline CoreVec3<T> Cross(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return CoreVec3<T>{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
    }

    // Distance ops
    template <typename T>
    inline T Distance(const CoreVec2<T>& a, const CoreVec2<T>& b)
    {
        return Length(a - b);
    }
    template <typename T>
    inline T Distance(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return Length(a - b);
    }
    template <typename T>
    inline T Distance(const CoreVec4<T>& a, const CoreVec4<T>& b)
    {
        return Length(a - b);
    }

    // Min ops
    template <typename T>
    inline CoreVec2<T> Min(const CoreVec2<T>& a, const CoreVec2<T>& b)
    {
        return CoreVec2<T>{ std::min(a.x, b.x), std::min(a.y, b.y) };
    }
    template <typename T>
    inline CoreVec3<T> Min(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return CoreVec3<T>{ std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) };
    }
    template <typename T>
    inline CoreVec4<T> Min(const CoreVec4<T>& a, const CoreVec4<T>& b)
    {
        return CoreVec4<T>{ std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w) };
    }

    // Max ops
    template <typename T>
    inline CoreVec2<T> Max(const CoreVec2<T>& a, const CoreVec2<T>& b)
    {
        return CoreVec2<T>{ std::max(a.x, b.x), std::max(a.y, b.y) };
    }
    template <typename T>
    inline CoreVec3<T> Max(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return CoreVec3<T>{ std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) };
    }
    template <typename T>
    inline CoreVec4<T> Max(const CoreVec4<T>& a, const CoreVec4<T>& b)
    {
        return CoreVec4<T>{ std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w) };
    }

} // namespace Ifrit::Math
