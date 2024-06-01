#pragma once
namespace Ifrit::Engine::TileRaster::CUDA {
	constexpr float CU_EPS = 1e-8f;

	// == Kernels ==
	constexpr int CU_LARGE_BIN_SIZE = 32;
	constexpr int CU_BIN_SIZE = 64;
	constexpr int CU_TILE_SIZE = 128;
	
	constexpr int CU_TILES_PER_BIN = CU_TILE_SIZE / CU_BIN_SIZE;
	constexpr int CU_BINS_PER_LARGE_BIN = CU_BIN_SIZE / CU_LARGE_BIN_SIZE;
	constexpr int CU_SUBTILE_SIZE_LOG = 2;
	constexpr int CU_SUBTILE_SIZE = (1<<(CU_SUBTILE_SIZE_LOG));
	constexpr int CU_MAX_VARYINGS = 10;
	constexpr int CU_MAX_ATTRIBUTES = 10;
	constexpr int CU_MAX_COVER_QUEUE_SIZE = 1440000;
	constexpr int CU_GEOMETRY_PROCESSING_THREADS = 64;
	constexpr int CU_RASTERIZATION_THREADS_PERDIM = 8;
	constexpr int CU_VERTEX_PROCESSING_THREADS = 64;

	constexpr int CU_FRAGMENT_SHADING_THREADS_PER_TILE_X = 10;
	constexpr int CU_FRAGMENT_SHADING_THREADS_PER_TILE_Y = 10;
	constexpr int CU_RASTERIZATION_THREADS_PER_TILE = 128;

	constexpr int CU_SINGLE_TIME_TRIANGLE = 20608; //Safe 20608 84480
	constexpr int CU_SINGLE_TIME_TRIANGLE_GEOMETRY_BATCHSIZE = 1;

	constexpr int CU_TRIANGLE_STRIDE = 3;
	constexpr float CU_LARGE_TRIANGLE_THRESHOLD = 0.15f;

	// == Memory Allocation ==
	constexpr size_t CU_HEAP_MEMORY_SIZE = 1024ull * 1024 * 1024 * 4;
	constexpr int CU_VECTOR_BASE_LENGTH = 11;
	constexpr int CU_VECTOR_HIERARCHY_LEVEL = 10;

	// == Options ==
	constexpr bool CU_OPT_HOMOGENEOUS_DISCARD = true;
	constexpr bool CU_OPT_HOMOGENEOUS_CLIPPING_NEG_W_ONLY = true;
	constexpr bool CU_OPT_CUDA_PROFILE = true;
	constexpr bool CU_OPT_PREALLOCATED_TRIANGLE_LIST = true;
}