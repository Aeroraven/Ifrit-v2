
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
