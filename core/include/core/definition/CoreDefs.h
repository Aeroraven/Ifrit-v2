#pragma once

#ifdef WIN32 
	//Note that this will influence the C# binding
	#define IFRIT_APICALL __stdcall
#else
	#define IFRIT_APICALL
#endif

#if _WINDLL
	#define IFRIT_DLL
	#define IFRIT_API_EXPORT
#endif

// Platform specific dllexport semantics
// https://stackoverflow.com/questions/2164827/explicitly-exporting-shared-library-functions-in-linux
#if defined(_MSC_VER)
	#define IFRIT_DLLEXPORT __declspec(dllexport)
	#define IFRIT_DLLIMPORT __declspec(dllimport)
#elif defined(__MINGW64__)
	#define IFRIT_DLLEXPORT __declspec(dllexport)
	#define IFRIT_DLLIMPORT __declspec(dllimport)
#elif defined(__clang__)
	#define IFRIT_DLLEXPORT __attribute__((visibility("default")))
	#define IFRIT_DLLIMPORT
#elif defined(__GNUC__)
	#define IFRIT_DLLEXPORT __attribute__((visibility("default")))
	#define IFRIT_DLLIMPORT
#else
	static_assert(false, "Unsupported compiler");
#endif


// x64 & x32 platform detection
#if _WIN32 || _WIN64
	#if _WIN64
		#define IFRIT_ENV64
	#else
		#define IFRIT_ENV32
		static_assert(false, "Lacking x32 support");
	#endif
#elif __GNUC__
	#if __x86_64__ || __ppc64__
		#define IFRIT_ENV64
	#else
		#define IFRIT_ENV32
		static_assert(false, "Lacking x32 support");
	#endif
#endif

#ifdef IFRIT_DLL
	#ifndef __cplusplus
		#define IFRIT_API_EXPORT_COMPATIBLE_MODE
	#endif // !__cplusplus

	#ifdef IFRIT_API_EXPORT_COMPATIBLE_MODE
		#ifdef IFRIT_API_EXPORT
			#define IFRIT_APIDECL
			#define IFRIT_APIDECL_FORCED  IFRIT_DLLEXPORT
			#define IFRIT_APIDECL_COMPAT extern "C" IFRIT_DLLEXPORT
		#else
			#define IFRIT_APIDECL 
			#define IFRIT_APIDECL_FORCED IFRIT_DLLIMPORT
			#define IFRIT_APIDECL_COMPAT extern "C" IFRIT_DLLIMPORT
			#define IRTIT_IGNORE_PRESENTATION_DEPS
		#endif
	#else 
		#ifdef IFRIT_API_EXPORT
			#define IFRIT_APIDECL IFRIT_DLLEXPORT
			#define IFRIT_APIDECL_FORCED  IFRIT_DLLEXPORT
			#define IFRIT_APIDECL_COMPAT extern "C" IFRIT_DLLEXPORT
		#else
			#define IFRIT_APIDECL IFRIT_DLLIMPORT
			#define IFRIT_APIDECL_FORCED IFRIT_DLLIMPORT
			#define IFRIT_APIDECL_COMPAT extern "C" IFRIT_DLLIMPORT
			#define IRTIT_IGNORE_PRESENTATION_DEPS
		#endif
	#endif
#else
	#define IFRIT_APIDECL_FORCED IFRIT_DLLEXPORT
	#define IFRIT_APIDECL
	#define IFRIT_APIDECL_COMPAT
#endif

#include <cstddef>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <optional>
#include <functional>
#include <fstream>
#include <ctime>
#include <regex>
#include <concepts>
#include <cmath>
#include <cstring>
#include <random>
#include <array>
#include <map>
#include <unordered_map>
#include <set>
#include <variant>
#include <unordered_set>
#include <chrono>
#include <ranges>
#include <typeinfo>
#include <sstream>
#include <deque>
#include <mutex>
#include <thread>

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
	//Patch for gcc10
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
	static_assert(false, "App requires C++20 or higher (Current compiler does not support C++17)");
#endif
#ifndef IFRIT_CXX20_ENABLED
	static_assert(false, "App requires C++20 or higher");
#endif

#ifdef IFRIT_CXX23_ENABLED
#include <print>
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
		static_assert(false, "It's assumed that machine with AVX2 support, must support SSE instructions. However SSE tag is not set.");
	#endif
#endif


#if IFRIT_FEATURE_CUDA
	#define IFRIT_USE_CUDA 1
	// Intellisense for cuda
	#ifdef __INTELLISENSE__
		#define CU_KARG2(grid, block)
		#define CU_KARG3(grid, block, sh_mem)
		#define CU_KARG4(grid, block, sh_mem, stream)
		#define __CUDACC__
		#define __THROW
		#define __CUDA_ARCH__
		#include <device_functions.h>
	#else
		#ifdef __CUDACC__
			#define CU_KARG2(grid, block) <<<grid, block>>>
			#define CU_KARG3(grid, block, sh_mem) <<<grid, block, sh_mem>>>
			#define CU_KARG4(grid, block, sh_mem, stream) <<<grid, block, sh_mem, stream>>>
		#else
			#define CU_KARG2(grid, block)
			#define CU_KARG3(grid, block, sh_mem)
			#define CU_KARG4(grid, block, sh_mem, stream)
		#endif
	#endif
	#include <cuda_runtime.h>
	#define IFRIT_DEVICE_CONST __constant__
	#define IFRIT_DEVICE __device__
	#define IFRIT_HOST __host__
	#define IFRIT_DUAL __host__ __device__
	#define IFRIT_KERNEL __global__
	#define IFRIT_SHARED __shared__
#else
	#define IFRIT_DUAL
	#define IFRIT_KERNEL
	#define IFRIT_DEVICE
	#define IFRIT_HOST
	#define IFRIT_SHARED
#endif

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