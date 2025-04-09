
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
#include "../base/IfritBase.h"
#include "../platform/ApiConv.h"
#include "VectorDefs.h"
#include <algorithm>
#include <cmath>

namespace Ifrit::Math
{
// Element wise ops
#define ELEMENTWISE_VECTOR_OP(op)                                                                                 \
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> operator op(const CoreVec2<T>& a, const CoreVec2<T>& b) \
    {                                                                                                             \
        return CoreVec2<T>{ a.x op b.x, a.y op b.y };                                                             \
    }                                                                                                             \
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> operator op(const CoreVec3<T>& a, const CoreVec3<T>& b) \
    {                                                                                                             \
        return CoreVec3<T>{ a.x op b.x, a.y op b.y, a.z op b.z };                                                 \
    }                                                                                                             \
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> operator op(const CoreVec4<T>& a, const CoreVec4<T>& b) \
    {                                                                                                             \
        return CoreVec4<T>{ a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w };                                     \
    }
    ELEMENTWISE_VECTOR_OP(+);
    ELEMENTWISE_VECTOR_OP(-);
    ELEMENTWISE_VECTOR_OP(*);
    ELEMENTWISE_VECTOR_OP(/);
#undef ELEMENTWISE_VECTOR_OP

// Scalar ops
#define SCALAR_VECTOR_OP(op)                                                                     \
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> operator op(const CoreVec2<T>& a, T b) \
    {                                                                                            \
        return CoreVec2<T>{ a.x op b, a.y op b };                                                \
    }                                                                                            \
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> operator op(const CoreVec3<T>& a, T b) \
    {                                                                                            \
        return CoreVec3<T>{ a.x op b, a.y op b, a.z op b };                                      \
    }                                                                                            \
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> operator op(const CoreVec4<T>& a, T b) \
    {                                                                                            \
        return CoreVec4<T>{ a.x op b, a.y op b, a.z op b, a.w op b };                            \
    }
    SCALAR_VECTOR_OP(+);
    SCALAR_VECTOR_OP(-);
    SCALAR_VECTOR_OP(*);
    SCALAR_VECTOR_OP(/);
#undef SCALAR_VECTOR_OP

#define SCALAR_VECTOR_OP_REV(op)                                                                 \
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> operator op(T b, const CoreVec2<T>& a) \
    {                                                                                            \
        return CoreVec2<T>{ b op a.x, b op a.y };                                                \
    }                                                                                            \
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> operator op(T b, const CoreVec3<T>& a) \
    {                                                                                            \
        return CoreVec3<T>{ b op a.x, b op a.y, b op a.z };                                      \
    }                                                                                            \
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> operator op(T b, const CoreVec4<T>& a) \
    {                                                                                            \
        return CoreVec4<T>{ b op a.x, b op a.y, b op a.z, b op a.w };                            \
    }

    SCALAR_VECTOR_OP_REV(+);
    SCALAR_VECTOR_OP_REV(-);
    SCALAR_VECTOR_OP_REV(*);
    SCALAR_VECTOR_OP_REV(/);

#undef SCALAR_VECTOR_OP_REV

    // Element wise ops 2

#define ELEMENTWISE_VECTOR_OP2(op)                                                              \
    template <typename T> inline CoreVec2<T>& operator op(CoreVec2<T>& a, const CoreVec2<T>& b) \
    {                                                                                           \
        a.x op b.x;                                                                             \
        a.y op b.y;                                                                             \
        return a;                                                                               \
    }                                                                                           \
    template <typename T> inline CoreVec3<T>& operator op(CoreVec3<T>& a, const CoreVec3<T>& b) \
    {                                                                                           \
        a.x op b.x;                                                                             \
        a.y op b.y;                                                                             \
        a.z op b.z;                                                                             \
        return a;                                                                               \
    }                                                                                           \
    template <typename T> inline CoreVec4<T>& operator op(CoreVec4<T>& a, const CoreVec4<T>& b) \
    {                                                                                           \
        a.x op b.x;                                                                             \
        a.y op b.y;                                                                             \
        a.z op b.z;                                                                             \
        a.w op b.w;                                                                             \
        return a;                                                                               \
    }
    ELEMENTWISE_VECTOR_OP2(+=);
    ELEMENTWISE_VECTOR_OP2(-=);
    ELEMENTWISE_VECTOR_OP2(*=);
    ELEMENTWISE_VECTOR_OP2(/=);

#undef ELEMENTWISE_VECTOR_OP2

// Scalar ops 2
#define SCALAR_VECTOR_OP2(op)                                                  \
    template <typename T> inline CoreVec2<T>& operator op(CoreVec2<T>& a, T b) \
    {                                                                          \
        a.x op b;                                                              \
        a.y op b;                                                              \
        return a;                                                              \
    }                                                                          \
    template <typename T> inline CoreVec3<T>& operator op(CoreVec3<T>& a, T b) \
    {                                                                          \
        a.x op b;                                                              \
        a.y op b;                                                              \
        a.z op b;                                                              \
        return a;                                                              \
    }                                                                          \
    template <typename T> inline CoreVec4<T>& operator op(CoreVec4<T>& a, T b) \
    {                                                                          \
        a.x op b;                                                              \
        a.y op b;                                                              \
        a.z op b;                                                              \
        a.w op b;                                                              \
        return a;                                                              \
    }

    SCALAR_VECTOR_OP2(+=);
    SCALAR_VECTOR_OP2(-=);
    SCALAR_VECTOR_OP2(*=);
    SCALAR_VECTOR_OP2(/=);

#undef SCALAR_VECTOR_OP2

    // Address ops
    template <typename T> inline IF_CONSTEXPR T ElementAt(CoreVec2<T>& a, int i) { return i == 0 ? a.x : a.y; }
    template <typename T> inline IF_CONSTEXPR T ElementAt(CoreVec3<T>& a, int i)
    {
        return i == 0 ? a.x : i == 1 ? a.y : a.z;
    }
    template <typename T> inline IF_CONSTEXPR T ElementAt(CoreVec4<T>& a, int i)
    {
        return i == 0 ? a.x : i == 1 ? a.y : i == 2 ? a.z : a.w;
    }

    // Normalize ops
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> Normalize(const CoreVec2<T>& a)
    {
        T len = sqrt(a.x * a.x + a.y * a.y);
        return CoreVec2<T>{ a.x / len, a.y / len };
    }
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> Normalize(const CoreVec3<T>& a)
    {
        T len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        return CoreVec3<T>{ a.x / len, a.y / len, a.z / len };
    }
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> Normalize(const CoreVec4<T>& a)
    {
        T len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
        return CoreVec4<T>{ a.x / len, a.y / len, a.z / len, a.w / len };
    }

    // Length ops
    template <typename T> inline IF_CONSTEXPR T Length(const CoreVec2<T>& a) { return sqrt(a.x * a.x + a.y * a.y); }
    template <typename T> inline IF_CONSTEXPR T Length(const CoreVec3<T>& a)
    {
        return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    }
    template <typename T> inline IF_CONSTEXPR T Length(const CoreVec4<T>& a)
    {
        return sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }

    // Dot ops
    template <typename T> inline IF_CONSTEXPR T Dot(const CoreVec2<T>& a, const CoreVec2<T>& b)
    {
        return a.x * b.x + a.y * b.y;
    }
    template <typename T> inline IF_CONSTEXPR T Dot(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    template <typename T> inline IF_CONSTEXPR T Dot(const CoreVec4<T>& a, const CoreVec4<T>& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // Lerp ops
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> Lerp(const CoreVec2<T>& a, const CoreVec2<T>& b, const T& t)
    {
        return a * (1 - t) + b * t;
    }
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> Lerp(const CoreVec3<T>& a, const CoreVec3<T>& b, const T& t)
    {
        return a * (1 - t) + b * t;
    }
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> Lerp(const CoreVec4<T>& a, const CoreVec4<T>& b, const T& t)
    {
        return a * (1 - t) + b * t;
    }

    // Clamp ops
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> Clamp(const CoreVec2<T>& a, const T& min, const T& max)
    {
        return CoreVec2<T>{ std::clamp(a.x, min, max), std::clamp(a.y, min, max) };
    }
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> Clamp(const CoreVec3<T>& a, const T& min, const T& max)
    {
        return CoreVec3<T>{ std::clamp(a.x, min, max), std::clamp(a.y, min, max), std::clamp(a.z, min, max) };
    }
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> Clamp(const CoreVec4<T>& a, const T& min, const T& max)
    {
        return CoreVec4<T>{ std::clamp(a.x, min, max), std::clamp(a.y, min, max), std::clamp(a.z, min, max),
            std::clamp(a.w, min, max) };
    }

    // Cross ops
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> Cross(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return CoreVec3<T>{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
    }

    // Distance ops
    template <typename T> inline IF_CONSTEXPR T Distance(const CoreVec2<T>& a, const CoreVec2<T>& b)
    {
        return Length(a - b);
    }
    template <typename T> inline IF_CONSTEXPR T Distance(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return Length(a - b);
    }
    template <typename T> inline IF_CONSTEXPR T Distance(const CoreVec4<T>& a, const CoreVec4<T>& b)
    {
        return Length(a - b);
    }

    // Min ops
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> Min(const CoreVec2<T>& a, const CoreVec2<T>& b)
    {
        return CoreVec2<T>{ std::min(a.x, b.x), std::min(a.y, b.y) };
    }
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> Min(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return CoreVec3<T>{ std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) };
    }
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> Min(const CoreVec4<T>& a, const CoreVec4<T>& b)
    {
        return CoreVec4<T>{ std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w) };
    }

    // Max ops
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> Max(const CoreVec2<T>& a, const CoreVec2<T>& b)
    {
        return CoreVec2<T>{ std::max(a.x, b.x), std::max(a.y, b.y) };
    }
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> Max(const CoreVec3<T>& a, const CoreVec3<T>& b)
    {
        return CoreVec3<T>{ std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) };
    }
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> Max(const CoreVec4<T>& a, const CoreVec4<T>& b)
    {
        return CoreVec4<T>{ std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w) };
    }

// Other Elementwise Functiosns
#define ELEMENTWISE_FUNC(funcName, func)                                                 \
    template <typename T> inline IF_CONSTEXPR CoreVec2<T> funcName(const CoreVec2<T>& a) \
    {                                                                                    \
        return CoreVec2<T>{ func(a.x), func(a.y) };                                      \
    }                                                                                    \
    template <typename T> inline IF_CONSTEXPR CoreVec3<T> funcName(const CoreVec3<T>& a) \
    {                                                                                    \
        return CoreVec3<T>{ func(a.x), func(a.y), func(a.z) };                           \
    }                                                                                    \
    template <typename T> inline IF_CONSTEXPR CoreVec4<T> funcName(const CoreVec4<T>& a) \
    {                                                                                    \
        return CoreVec4<T>{ func(a.x), func(a.y), func(a.z), func(a.w) };                \
    }

    ELEMENTWISE_FUNC(Sqrt, std::sqrt);
    ELEMENTWISE_FUNC(Abs, std::abs);
    ELEMENTWISE_FUNC(Cos, std::cos);
    ELEMENTWISE_FUNC(Sin, std::sin);
    ELEMENTWISE_FUNC(Tan, std::tan);
    ELEMENTWISE_FUNC(Asin, std::asin);
    ELEMENTWISE_FUNC(Acos, std::acos);
    ELEMENTWISE_FUNC(Atan, std::atan);
    ELEMENTWISE_FUNC(Exp, std::exp);
    ELEMENTWISE_FUNC(Log, std::log);
    ELEMENTWISE_FUNC(Log2, std::log2);
    ELEMENTWISE_FUNC(Log10, std::log10);
    ELEMENTWISE_FUNC(Ceil, std::ceil);
    ELEMENTWISE_FUNC(Floor, std::floor);
    ELEMENTWISE_FUNC(Round, std::round);
    ELEMENTWISE_FUNC(Fract, std::modf);
    ELEMENTWISE_FUNC(Pow, std::pow);
    ELEMENTWISE_FUNC(Sign, std::signbit);

    // Logical Ops
    template <typename T> inline IF_CONSTEXPR bool Any(const CoreVec2<T>& a) { return a.x || a.y; }
    template <typename T> inline IF_CONSTEXPR bool Any(const CoreVec3<T>& a) { return a.x || a.y || a.z; }
    template <typename T> inline IF_CONSTEXPR bool Any(const CoreVec4<T>& a) { return a.x || a.y || a.z || a.w; }

    template <typename T> inline IF_CONSTEXPR bool All(const CoreVec2<T>& a) { return a.x && a.y; }
    template <typename T> inline IF_CONSTEXPR bool All(const CoreVec3<T>& a) { return a.x && a.y && a.z; }
    template <typename T> inline IF_CONSTEXPR bool All(const CoreVec4<T>& a) { return a.x && a.y && a.z && a.w; }

    template <typename T> inline IF_CONSTEXPR bool None(const CoreVec2<T>& a) { return !a.x && !a.y; }
    template <typename T> inline IF_CONSTEXPR bool None(const CoreVec3<T>& a) { return !a.x && !a.y && !a.z; }
    template <typename T> inline IF_CONSTEXPR bool None(const CoreVec4<T>& a) { return !a.x && !a.y && !a.z && !a.w; }

    template <typename T> inline bool ContainsNaN(const CoreVec2<T>& a) { return std::isnan(a.x) || std::isnan(a.y); }
    template <typename T> inline bool ContainsNaN(const CoreVec3<T>& a)
    {
        return std::isnan(a.x) || std::isnan(a.y) || std::isnan(a.z);
    }
    template <typename T> inline bool ContainsNaN(const CoreVec4<T>& a)
    {
        return std::isnan(a.x) || std::isnan(a.y) || std::isnan(a.z) || std::isnan(a.w);
    }

// Equal
#define COMPARE_VECTOR_OP(op)                                                                                        \
    template <typename T> inline IF_CONSTEXPR CoreVec2<bool> operator op(const CoreVec2<T>& a, const CoreVec2<T>& b) \
    {                                                                                                                \
        return CoreVec2<bool>{ a.x op b.x ? true : false, a.y op b.y ? true : false };                               \
    }                                                                                                                \
    template <typename T> inline IF_CONSTEXPR CoreVec3<bool> operator op(const CoreVec3<T>& a, const CoreVec3<T>& b) \
    {                                                                                                                \
        return CoreVec3<bool>{ a.x op b.x ? true : false, a.y op b.y ? true : false, a.z op b.z ? true : false };    \
    }                                                                                                                \
    template <typename T> inline IF_CONSTEXPR CoreVec4<bool> operator op(const CoreVec4<T>& a, const CoreVec4<T>& b) \
    {                                                                                                                \
        return CoreVec4<bool>{ a.x op b.x ? true : false, a.y op b.y ? true : false, a.z op b.z ? true : false,      \
            a.w op b.w ? true : 0 };                                                                                 \
    }

    COMPARE_VECTOR_OP(==);
    COMPARE_VECTOR_OP(!=);
    COMPARE_VECTOR_OP(<);
    COMPARE_VECTOR_OP(<=);
    COMPARE_VECTOR_OP(>);
    COMPARE_VECTOR_OP(>=);

#undef COMPARE_VECTOR_OP

#define COMPARE_VECTOR_SCALAR_OP(op)                                                                        \
    template <typename T> inline IF_CONSTEXPR CoreVec2<bool> operator op(const CoreVec2<T>& a, T b)         \
    {                                                                                                       \
        return CoreVec2<bool>{ a.x op b ? true : false, a.y op b ? true : false };                          \
    }                                                                                                       \
    template <typename T> inline IF_CONSTEXPR CoreVec3<bool> operator op(const CoreVec3<T>& a, T b)         \
    {                                                                                                       \
        return CoreVec3<bool>{ a.x op b ? true : false, a.y op b ? true : false, a.z op b ? true : false }; \
    }                                                                                                       \
    template <typename T> inline IF_CONSTEXPR CoreVec4<bool> operator op(const CoreVec4<T>& a, T b)         \
    {                                                                                                       \
        return CoreVec4<bool>{ a.x op b ? true : false, a.y op b ? true : false, a.z op b ? true : false,   \
            a.w op b ? true : false };                                                                      \
    }

    COMPARE_VECTOR_SCALAR_OP(==);
    COMPARE_VECTOR_SCALAR_OP(!=);
    COMPARE_VECTOR_SCALAR_OP(<);
    COMPARE_VECTOR_SCALAR_OP(<=);
    COMPARE_VECTOR_SCALAR_OP(>);
    COMPARE_VECTOR_SCALAR_OP(>=);
#undef COMPARE_VECTOR_SCALAR_OP

#define COMPARE_SCALAR_VECTOR_OP(op)                                                                        \
    template <typename T> inline IF_CONSTEXPR CoreVec2<bool> operator op(T a, const CoreVec2<T>& b)         \
    {                                                                                                       \
        return CoreVec2<bool>{ a op b.x ? true : false, a op b.y ? true : false };                          \
    }                                                                                                       \
    template <typename T> inline IF_CONSTEXPR CoreVec3<bool> operator op(T a, const CoreVec3<T>& b)         \
    {                                                                                                       \
        return CoreVec3<bool>{ a op b.x ? true : false, a op b.y ? true : false, a op b.z ? true : false }; \
    }                                                                                                       \
    template <typename T> inline IF_CONSTEXPR CoreVec4<bool> operator op(T a, const CoreVec4<T>& b)         \
    {                                                                                                       \
        return CoreVec4<bool>{ a op b.x ? true : false, a op b.y ? true : false, a op b.z ? true : false,   \
            a op b.w ? true : 0 };                                                                          \
    }

    COMPARE_SCALAR_VECTOR_OP(==);
    COMPARE_SCALAR_VECTOR_OP(!=);
    COMPARE_SCALAR_VECTOR_OP(<);
    COMPARE_SCALAR_VECTOR_OP(<=);
    COMPARE_SCALAR_VECTOR_OP(>);
    COMPARE_SCALAR_VECTOR_OP(>=);
#undef COMPARE_SCALAR_VECTOR_OP

} // namespace Ifrit::Math
