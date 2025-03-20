
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

#include <cstdint>
#ifdef __cplusplus
#include <memory>
#include <vector>
#endif

#define IF_SIZEOF_RETURN_TYPE u32

// check if compiler supports constexpr
#if _MSC_VER >= 1920
#define IF_CONSTEXPR constexpr
#define IF_CONSTEXPR_AVAILABLE 1
#else
#if __cplusplus >= 201703L
#define IF_CONSTEXPR constexpr
#define IF_CONSTEXPR_AVAILABLE 1
#else
#define IF_CONSTEXPR
#endif
#endif

// check if compiler support noexcept
#if _MSC_VER >= 1900
#define IF_NOEXCEPT noexcept
#else
#if __cplusplus >= 201703L
#define IF_NOEXCEPT noexcept
#else
#define IF_NOEXCEPT
#endif
#endif

// forceinline
#if _MSC_VER >= 1900
#define IF_FORCEINLINE __forceinline
#else
#if __cplusplus >= 201703L
#define IF_FORCEINLINE inline
#else
#define IF_FORCEINLINE
#endif
#endif

namespace Ifrit {

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef intptr_t isize;
typedef uintptr_t usize;

#define IF_TYPE_SIZEOF(type) (static_cast<IF_SIZEOF_RETURN_TYPE>(sizeof(type)))

// if have constexpr, use it
#if defined(IF_CONSTEXPR_AVAILABLE)
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE u8Size = IF_TYPE_SIZEOF(u8);
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE u16Size = IF_TYPE_SIZEOF(u16);
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE u32Size = IF_TYPE_SIZEOF(u32);
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE u64Size = IF_TYPE_SIZEOF(u64);

IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE i8Size = IF_TYPE_SIZEOF(i8);
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE i16Size = IF_TYPE_SIZEOF(i16);
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE i32Size = IF_TYPE_SIZEOF(i32);
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE i64Size = IF_TYPE_SIZEOF(i64);

IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE f32Size = IF_TYPE_SIZEOF(f32);
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE f64Size = IF_TYPE_SIZEOF(f64);

IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE isizeSize = IF_TYPE_SIZEOF(isize);
IF_CONSTEXPR IF_SIZEOF_RETURN_TYPE usizeSize = IF_TYPE_SIZEOF(usize);

#else
// if not, use normal variable
#define u8Size IF_TYPE_SIZEOF(u8)
#define u16Size IF_TYPE_SIZEOF(u16)
#define u32Size IF_TYPE_SIZEOF(u32)
#define u64Size IF_TYPE_SIZEOF(u64)

#define i8Size IF_TYPE_SIZEOF(i8)
#define i16Size IF_TYPE_SIZEOF(i16)
#define i32Size IF_TYPE_SIZEOF(i32)
#define i64Size IF_TYPE_SIZEOF(i64)

#define f32Size IF_TYPE_SIZEOF(f32)
#define f64Size IF_TYPE_SIZEOF(f64)

#define isizeSize IF_TYPE_SIZEOF(isize)
#define usizeSize IF_TYPE_SIZEOF(usize)
#endif

#ifdef __cplusplus
template <typename T> using Vec = std::vector<T>;
template <typename T> using Ref = std::shared_ptr<T>;
template <typename T> using Uref = std::unique_ptr<T>;
#endif

} // namespace Ifrit
