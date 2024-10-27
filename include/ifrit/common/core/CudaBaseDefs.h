#pragma once

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
