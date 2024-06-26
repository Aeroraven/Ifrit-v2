#pragma once
#include <memory>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <optional>
#include <set>
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
#include <any>

#ifdef _HAS_CXX23
	#define IFRIT_CXX23_ENABLED 1
#endif
#ifndef _HAS_CXX23
	#if __cplusplus >= 202302L
		#define IFRIT_CXX23_ENABLED 1
	#endif
#endif

#ifdef _HAS_CXX20
	#define IFRIT_CXX20_ENABLED 1
#endif
#ifndef _HAS_CXX20
	#if __cplusplus >= 202002L
		#define IFRIT_CXX20_ENABLED 1
	#endif
#endif

#ifdef _HAS_CXX17
	#define IFRIT_CXX17_ENABLED 1
#endif
#ifndef _HAS_CXX17
	static_assert(false, "App requires C++17 or higher")
#endif

#ifdef IFRIT_CXX23_ENABLED
#include <print>
#endif 

#ifndef __INTELLISENSE__
	#ifndef IFRIT_SHADER_PATH
		static_assert(false, "IFRIT_SHADER_PATH is not defined");
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
	#include <immintrin.h>
	#define IFRIT_USE_SIMD_256 1
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