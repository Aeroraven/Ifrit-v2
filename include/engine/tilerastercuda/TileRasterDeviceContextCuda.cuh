#pragma once
#include "core/definition/CoreDefs.h"
#include "core/cuda/CudaUtils.cuh"
#include "engine/tileraster/TileRasterCommon.h"
#include "engine/base/VaryingStore.h"
#include <vector>

namespace Ifrit::Engine::TileRaster::CUDA {
	using Ifrit::Core::CUDA::DeviceVector;
	using namespace Ifrit::Engine::TileRaster;
	using namespace Ifrit::Engine;

	struct TileRasterDeviceContext {
		std::vector<DeviceVector<VaryingStore>> hdVaryingBuffer;
		std::vector<VaryingStore*> hdVaryingBufferVec;

		std::vector<DeviceVector<AssembledTriangleProposal>> hdAssembledTriangles;
		std::vector<AssembledTriangleProposal*> hdAssembledTrianglesVec;

		std::vector<DeviceVector<TileBinProposal>> hdRasterQueue;
		std::vector<TileBinProposal*> hdRasterQueueVec;

		std::vector<DeviceVector<TileBinProposal>> hdCoverQueue;
		std::vector<TileBinProposal*> hdCoverQueueVec;
	};
	
}