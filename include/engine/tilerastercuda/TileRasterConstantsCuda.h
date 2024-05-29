#pragma once
namespace Ifrit::Engine::TileRaster::CUDA {
	constexpr float CU_EPS = 1e-8f;
	constexpr int CU_TILE_SIZE = 128;
	constexpr int CU_TILE_FIRST_STEP = 4;
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

	constexpr int CU_SINGLE_TIME_TRIANGLE = 20608;
	constexpr int CU_SINGLE_TIME_TRIANGLE_GEOMETRY_BATCHSIZE = 1;

	constexpr int CU_TRIANGLE_STRIDE = 3;

	constexpr bool CU_OPT_HOMOGENEOUS_CLIPPING_NEG_W_ONLY = true;
}