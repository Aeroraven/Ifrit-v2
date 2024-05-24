#pragma once
namespace Ifrit::Engine::TileRaster::CUDA {
	constexpr float CU_EPS = 5e-8f;
	constexpr int CU_TILE_SIZE = 128;
	constexpr int CU_SUBTILE_SIZE = 2;
	constexpr int CU_MAX_VARYINGS = 10;
	constexpr int CU_MAX_ATTRIBUTES = 10;
	constexpr int CU_MAX_COVER_QUEUE_SIZE = 1280000;
	constexpr int CU_GEOMETRY_PROCESSING_THREADS = 32;
	constexpr int CU_RASTERIZATION_THREADS_PERDIM = 8;
	constexpr int CU_VERTEX_PROCESSING_THREADS = 32;
	constexpr int CU_RASTERIZATION_THREADS = 256;

	constexpr int CU_FRAGMENT_SHADING_THREADS_PER_TILE_X = 8;
	constexpr int CU_FRAGMENT_SHADING_THREADS_PER_TILE_Y = 8;
	constexpr int CU_RASTERIZATION_THREADS_PER_TILE = 32;

	constexpr bool CU_FRAGMENT_LAUCH_SUBKERNEL = false;
	constexpr int CU_SINGLE_TIME_TRIANGLE = 7200;
	constexpr int CU_TRIANGLE_STRIDE = 3;
	constexpr int CU_FRAGMENT_CANDPROC_PER_LOOP_PER_THREAD = 4;
	constexpr int CU_FRAGMENT_CANDPROC_PER_LOOP = CU_FRAGMENT_SHADING_THREADS_PER_TILE_X * CU_FRAGMENT_SHADING_THREADS_PER_TILE_Y * CU_FRAGMENT_CANDPROC_PER_LOOP_PER_THREAD;

}