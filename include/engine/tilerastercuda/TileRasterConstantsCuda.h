#pragma once
namespace Ifrit::Engine::TileRaster::CUDA {
	constexpr float CU_EPS = 5e-9f;

	// == Device ==
	constexpr int CU_WARP_SIZE = 32;

	// == Kernels ==
	constexpr int CU_LARGE_BIN_SIZE = 32;
	constexpr int CU_BIN_SIZE = 128;
	constexpr int CU_TILE_SIZE = 128;
	constexpr int CU_MAX_FRAMEBUFFER_SIZE = 4096 * 4096;
	constexpr int CU_MAX_TEXTURE_SLOTS = 16;

	// experimentals
	constexpr int CU_LARGE_BIN_WIDTH = 64;
	constexpr int CU_BIN_WIDTH = 16;
	constexpr int CU_TILE_WIDTH = 16;

	constexpr int CU_MAX_BIN_X = 128;
	constexpr int CU_MAX_TILE_X = 128;
	constexpr int CU_MAX_LARGE_BIN_X = 32;
	// end experimentals
	
	constexpr int CU_TILES_PER_BIN = CU_TILE_SIZE / CU_BIN_SIZE;
	constexpr int CU_BINS_PER_LARGE_BIN = CU_BIN_SIZE / CU_LARGE_BIN_SIZE;
	constexpr int CU_SUBTILE_SIZE_LOG = 2;
	constexpr int CU_SUBTILE_SIZE = (1<<(CU_SUBTILE_SIZE_LOG));
	constexpr int CU_MAX_VARYINGS = 2;
	constexpr int CU_MAX_ATTRIBUTES = 2;
	constexpr int CU_GEOMETRY_PROCESSING_THREADS = 128;
	constexpr int CU_RASTERIZATION_THREADS_PERDIM = 8;
	constexpr int CU_VERTEX_PROCESSING_THREADS = 64;

	constexpr int CU_FRAGMENT_SHADING_THREADS_PER_TILE_X = 10;
	constexpr int CU_FRAGMENT_SHADING_THREADS_PER_TILE_Y = 10;
	constexpr int CU_FIRST_RASTERIZATION_THREADS_TINY = 64;
	constexpr int CU_FIRST_RASTERIZATION_THREADS = 16;
	constexpr int CU_FIRST_RASTERIZATION_LARGE_FINER_PROC = 16;
	constexpr int CU_FIRST_RASTERIZATION_GATHER_THREADS = 128;
	constexpr int CU_FIRST_RASTERIZATION_THREADS_LARGE = 8;

	constexpr int CU_SECOND_RASTERIZATION_THREADS_PER_TILE = 256;
	constexpr int CU_SECOND_RASTERIZATION_GATHER_THREADS = 128;

	constexpr int CU_PRIMITIVE_BUFFER_SIZE = 1154048 *2; //Safe 20608 84480
	constexpr int CU_SINGLE_TIME_TRIANGLE = 1154048 / 2; //Safe 20608 84480
	constexpr int CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER = 84480;

	constexpr int CU_ALTERNATIVE_BUFFER_SIZE = 654048;

	constexpr int CU_FIRST_BINNER_STRIDE_TINY = 2;
	constexpr int CU_FIRST_BINNER_STRIDE = 8;
	constexpr int CU_FIRST_BINNER_STRIDE_LARGE = 4;
	constexpr float CU_LARGE_TRIANGLE_THRESHOLD = 0.15f;
	constexpr int CU_MAX_SUBTILES_PER_TILE = 16;

	constexpr int CU_ELEMENTS_PER_SECOND_BINNER_BLOCK = 8;

	// == Memory Allocation ==
	constexpr size_t CU_HEAP_MEMORY_SIZE = 1024ull * 1024 * 1024 * 4;
	constexpr int CU_VECTOR_BASE_LENGTH = 9;
	constexpr int CU_VECTOR_HIERARCHY_LEVEL = 10;

	// == Options ==
	constexpr bool CU_OPT_HOMOGENEOUS_DISCARD = false;
	constexpr bool CU_OPT_CUDA_PROFILE = true;
	constexpr bool CU_OPT_PREALLOCATED_TRIANGLE_LIST = false;
	constexpr bool CU_OPT_II_SKIP_ON_FEW_GEOMETRIES = true;
	constexpr bool CU_OPT_ALIGNED_INDEX_BUFFER = true;  
	constexpr bool CU_OPT_SEPARATE_FIRST_BINNER_KERNEL = true;
	constexpr bool CU_OPT_SMALL_PRIMITIVE_CULL = true;

	// == Derived == 
	constexpr int CU_TRIANGLE_STRIDE = CU_OPT_ALIGNED_INDEX_BUFFER ? 4 : 3;

	// == Experimental ==
	constexpr int CU_EXPERIMENTAL_SUBTILE_WIDTH = 4;
	constexpr int CU_EXPERIMENTAL_PIXELS_PER_SUBTILE = CU_EXPERIMENTAL_SUBTILE_WIDTH * CU_EXPERIMENTAL_SUBTILE_WIDTH;

	constexpr int CU_EXPERIMENTAL_SECOND_BINNER_WORKLIST_THREADS = 128;
	constexpr int CU_EXPERIMENTAL_II_FEW_GEOMETRIES_LIMIT = CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;

	constexpr int CU_EXPERIMENTAL_GEOMETRY_POSTPROC_THREADS = 128;
	constexpr float CU_EXPERIMENTAL_TINY_THRESHOLD = 1e-3;
	constexpr float CU_EXPERIMENTAL_SMALL_PRIMITIVE_CULL_THRESHOLD = 5e-2;

	// == Profiler ==
	constexpr bool CU_PROFILER_OVERDRAW = false;
	constexpr bool CU_PROFILER_SECOND_BINNER_UTILIZATION = false;
	constexpr bool CU_PROFILER_TRIANGLE_SETUP = false;
	constexpr bool CU_PROFILER_SECOND_BINNER_WORKQUEUE = false;
	constexpr bool CU_PROFILER_SMALL_TRIANGLE_OVERHEAD = false;
	constexpr bool CU_PROFILER_II_CPU_NSIGHT = false;
	constexpr bool CU_PROFILER_ENABLE_MEMCPY = true;
}