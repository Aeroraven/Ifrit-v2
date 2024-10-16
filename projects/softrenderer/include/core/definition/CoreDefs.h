#pragma once


#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <ranges>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// Check if the compiler is GCC-family (GCC and MinGW)
#if defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__)
#define IFRIT_COMPILER_GCC 1
#endif

#if _HAS_CXX23
#define IFRIT_CXX23_ENABLED 1
#endif
#ifndef _HAS_CXX23
#if __cplusplus >= 202302L
#define IFRIT_CXX23_ENABLED 1
#endif
#endif

#if _HAS_CXX20
#define IFRIT_CXX20_ENABLED 1
#endif
#ifndef _HAS_CXX20
// Patch for gcc10
#ifdef __GNUC__
#if __GNUC__ >= 10
#define IFRIT_CXX20_ENABLED 1
#endif
#endif
#if __cplusplus >= 202002L
#define IFRIT_CXX20_ENABLED 1
#endif
#endif

#if _HAS_CXX17
#define IFRIT_CXX17_ENABLED 1
#endif
#ifndef _HAS_CXX17
#if __cplusplus >= 201703L
#define IFRIT_CXX17_ENABLED 1
#endif
#endif
#ifndef IFRIT_CXX17_ENABLED
static_assert(
    false,
    "App requires C++20 or higher (Current compiler does not support C++17)");
#endif
#ifndef IFRIT_CXX20_ENABLED
static_assert(false, "App requires C++20 or higher");
#endif

#ifdef IFRIT_CXX23_ENABLED
#include <print>
#endif

#ifdef IFRIT_DLL
#define IRTIT_IGNORE_PRESENTATION_DEPS
#endif

#ifndef __INTELLISENSE__
#ifndef IFRIT_SHADER_PATH
#ifndef IRTIT_IGNORE_PRESENTATION_DEPS
static_assert(false, "IFRIT_SHADER_PATH is not defined");
#endif
#endif
#endif

#define IFRIT_VERBOSE_SAFETY_CHECK

#if IFRIT_FEATURE_SIMD
#include <emmintrin.h>
#define IFRIT_USE_SIMD_128 1
#define IFRIT_USE_SIMD_128_EXPERIMENTAL 0
#endif

#if IFRIT_FEATURE_SIMD_AVX512
#include <immintrin.h>
#define IFRIT_USE_SIMD_512 1
#endif

#if IFRIT_FEATURE_SIMD_AVX256
#if IFRIT_FEATURE_SIMD
#include <immintrin.h>
#define IFRIT_USE_SIMD_256 1
#else
static_assert(false, "It's assumed that machine with AVX2 support, must "
                     "support SSE instructions. However SSE tag is not set.");
#endif
#endif

#include <common/core/ApiConv.h>
#include <common/core/CudaBaseDefs.h>

#ifdef IFRIT_CXX20_ENABLED
#define IFRIT_BRANCH_LIKELY [[likely]]
#define IFRIT_BRANCH_UNLIKELY [[unlikely]]
#else
#define IFRIT_BRANCH_LIKELY
#define IFRIT_BRANCH_UNLIKELY
#endif

#if IFRIT_FEATURE_AGGRESSIVE_PERFORMANCE
#define IFRIT_AP_NOTHROW noexcept
#define IFRIT_AP_RESTRICT __restrict
#else
#define IFRIT_AP_NOTHROW
#define IFRIT_AP_RESTRICT
#endif

#define IFRIT_DECLARE_VERTEX_SHADER
#define IFRIT_RESTRICT_CUDA __restrict__
#define IFRIT_ASSUME __assume
#define IFRIT_NOTHROW noexcept
#define IFRIT_EXPORT_COMPAT_NOTHROW IFRIT_NOTHROW