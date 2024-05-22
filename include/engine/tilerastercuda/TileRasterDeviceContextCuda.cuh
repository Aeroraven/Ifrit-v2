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

		TileBinProposal** dCoverQueue2;

		TileBinProposal* dCoverQueue;
		uint32_t* dCoverQueueCounter;
		uint32_t* dShadingQueue;
		uint32_t* dRasterQueueCounter;
		uint32_t* dAssembledTrianglesCounter;

		AssembledTriangleProposal** dAssembledTriangles;
		TileBinProposal** dRasterQueue;
		VaryingStore** dVaryingBuffer;
		TileRasterDeviceConstants* dDeviceConstants;
	};
}