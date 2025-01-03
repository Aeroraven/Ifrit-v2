
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
namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA {
constexpr float CU_EPS = 1e-8f;

// == Device ==
constexpr int CU_WARP_SIZE = 32;

// == Kernels ==
constexpr int CU_MAX_FRAMEBUFFER_WIDTH = 4096;
constexpr int CU_MAX_FRAMEBUFFER_SIZE = 4096 * 4096;
constexpr int CU_MAX_TEXTURE_SLOTS = 16;
constexpr int CU_MAX_SAMPLER_SLOTS = 16;
constexpr int CU_MAX_BUFFER_SLOTS = 16;

// experimentals
constexpr int CU_LARGE_BIN_WIDTH = 64;
constexpr int CU_BIN_WIDTH = 16;
constexpr int CU_TILE_WIDTH = 16;

constexpr int CU_MAX_BIN_X = 128;
constexpr int CU_MAX_TILE_X = 128;
constexpr int CU_MAX_LARGE_BIN_X = 32;
// end experimentals

// points
constexpr int CU_POINT_RASTERIZATION_FIRST_THREADS = 128;
constexpr int CU_POINT_RASTERIZATION_PLACE_THREADS = 128;
// end points

// lines
constexpr int CU_LINE_GEOMETRY_THREADS = 128;
constexpr int CU_LINE_RASTERIZATION_FIRST_THREADS = 128;
constexpr int CU_LINE_RASTERIZATION_PLACE_THREADS = 128;
// end lines

// sort
constexpr int CU_SORT_KEYGEN_THREADS = 128;
// end sort

constexpr int CU_TILES_PER_BIN = CU_BIN_WIDTH / CU_TILE_WIDTH;
constexpr int CU_BINS_PER_LARGE_BIN = CU_LARGE_BIN_WIDTH / CU_BIN_WIDTH;
constexpr int CU_SUBTILE_SIZE_LOG = 2;
constexpr int CU_SUBTILE_SIZE = (1 << (CU_SUBTILE_SIZE_LOG));
constexpr int CU_MAX_VARYINGS = 2;
constexpr int CU_MAX_ATTRIBUTES = 3;
constexpr int CU_MAX_GS_OUT_VERTICES = 3;
constexpr int CU_GEOMETRY_PROCESSING_THREADS = 128;
constexpr int CU_RASTERIZATION_THREADS_PERDIM = 8;
constexpr int CU_VERTEX_PROCESSING_THREADS = 96;

constexpr int CU_FRAGMENT_SHADING_THREADS_PER_TILE_X = 10;
constexpr int CU_FRAGMENT_SHADING_THREADS_PER_TILE_Y = 10;
constexpr int CU_FIRST_RASTERIZATION_THREADS_TINY = 64;
constexpr int CU_FIRST_RASTERIZATION_THREADS = 16;
constexpr int CU_FIRST_RASTERIZATION_LARGE_FINER_PROC = 16;
constexpr int CU_FIRST_RASTERIZATION_GATHER_THREADS = 128;
constexpr int CU_FIRST_RASTERIZATION_THREADS_LARGE = 8;

constexpr int CU_SECOND_RASTERIZATION_THREADS_PER_TILE = 256;
constexpr int CU_SECOND_RASTERIZATION_GATHER_THREADS = 128;

constexpr int CU_PRIMITIVE_BUFFER_SIZE = 1154048 * 2; // Safe 20608 84480
constexpr int CU_SINGLE_TIME_TRIANGLE = 1154048 / 2;  // Safe 20608 84480
constexpr int CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER = 84480;

constexpr int CU_ALTERNATIVE_BUFFER_SIZE_SECOND = 6254048;
constexpr int CU_ALTERNATIVE_BUFFER_SIZE = 1254048;

constexpr int CU_FIRST_BINNER_STRIDE_TINY = 2;
constexpr int CU_FIRST_BINNER_STRIDE = 8;
constexpr int CU_FIRST_BINNER_STRIDE_LARGE = 4;
constexpr float CU_LARGE_TRIANGLE_THRESHOLD = 0.15f;
constexpr int CU_MAX_SUBTILES_PER_TILE = 16;

constexpr int CU_ELEMENTS_PER_SECOND_BINNER_BLOCK = 8;
constexpr int CU_ELEMENTS_PER_FINER_SECOND_BINNER_BLOCK = 128;

constexpr int CU_GEOMETRY_SHADER_THREADS = 128;

// == Memory Allocation ==
constexpr size_t CU_HEAP_MEMORY_SIZE = 1024ull * 1024 * 1024 * 1;

// == Options ==
constexpr bool CU_OPT_SHADER_DERIVATIVES = true;
constexpr bool CU_OPT_HOMOGENEOUS_DISCARD = false;
constexpr bool CU_OPT_CUDA_PROFILE = true;
constexpr bool CU_OPT_II_SKIP_ON_FEW_GEOMETRIES = true;
constexpr bool CU_OPT_SMALL_PRIMITIVE_CULL = true;
constexpr bool CU_OPT_PATCH_STRICT_BOUNDARY = true;
constexpr bool CU_OPT_FORCE_DETERMINISTIC_BEHAVIOR = false;

// == Derived ==
constexpr int CU_TRIANGLE_STRIDE = 3;
constexpr int CU_GS_OUT_BUFFER_SIZE =
    CU_PRIMITIVE_BUFFER_SIZE * CU_MAX_GS_OUT_VERTICES;

// == Experimental ==
constexpr int CU_EXPERIMENTAL_SUBTILE_WIDTH = 4;
constexpr int CU_EXPERIMENTAL_PIXELS_PER_SUBTILE =
    CU_EXPERIMENTAL_SUBTILE_WIDTH * CU_EXPERIMENTAL_SUBTILE_WIDTH;

constexpr int CU_EXPERIMENTAL_SECOND_BINNER_WORKLIST_THREADS = 128;
constexpr int CU_EXPERIMENTAL_II_FEW_GEOMETRIES_LIMIT =
    CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;

constexpr int CU_EXPERIMENTAL_GEOMETRY_POSTPROC_THREADS = 128;
constexpr float CU_EXPERIMENTAL_TINY_THRESHOLD = 1e-3f;
constexpr float CU_EXPERIMENTAL_SMALL_PRIMITIVE_CULL_THRESHOLD = 5e-2f;

// == Profiler ==
constexpr bool CU_PROFILER_OVERDRAW = false;
constexpr bool CU_PROFILER_SECOND_BINNER_UTILIZATION = false;
constexpr bool CU_PROFILER_TRIANGLE_SETUP = false;
constexpr bool CU_PROFILER_SECOND_BINNER_WORKQUEUE = false;
constexpr bool CU_PROFILER_SMALL_TRIANGLE_OVERHEAD = false;
constexpr bool CU_PROFILER_SECOND_BINNER_THREAD_DIVERGENCE = true;
constexpr bool CU_PROFILER_II_CPU_NSIGHT = false;
constexpr bool CU_PROFILER_ENABLE_MEMCPY = true;

// == Ext: Mesh Shader ==
constexpr int CU_MESHSHADER_MAX_VERTICES = 256 * 3;
constexpr int CU_MESHSHADER_MAX_INDICES = 256 * 3;
constexpr int CU_MESHSHADER_MAX_WORKGROUPS = 740;
constexpr int CU_MESHSHADER_BUFFER_SIZE = 706432;

constexpr int CU_MESHSHADER_MAX_TASK_OUTPUT = 32;
constexpr int CU_MESHSHADER_MAX_TASK_PAYLOAD_SIZE = 128;

// == Ext: Scissor Test ==
constexpr bool CU_SCISSOR_ENABLE = true;
constexpr int CU_SCISSOR_MAX_COUNT = 16;

// == Ext: Patch Float Inaccuracy 240812 ==
constexpr bool CU_PATCH_FI_240812 = true;

// == Ext: MSAA ==
constexpr bool CU_MSAA_ENABLED = true;
constexpr int CU_MSAA_MAX_SAMPLES = 16;
} // namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA