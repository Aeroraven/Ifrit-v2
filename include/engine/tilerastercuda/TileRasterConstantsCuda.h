#pragma once
namespace Ifrit::Engine::TileRaster::CUDA {
	constexpr float CU_EPS = 1e-7f;
	constexpr int CU_TILE_SIZE = 64;
	constexpr int CU_SUBTILE_SIZE = 2;
	constexpr int CU_MAX_VARYINGS = 10;
	constexpr int CU_MAX_COVER_QUEUE_SIZE = 960000;
	constexpr int CU_GEOMETRY_PROCESSING_THREADS = 128;
	constexpr int CU_RASTERIZATION_THREADS_PERDIM = 8;
	constexpr int CU_VERTEX_PROCESSING_THREADS = 128;
	constexpr int CU_RASTERIZATION_THREADS = 256;
}