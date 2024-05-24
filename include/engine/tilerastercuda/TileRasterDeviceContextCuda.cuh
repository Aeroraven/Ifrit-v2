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

		std::vector<DeviceVector<uint32_t>> hdRasterQueue;
		std::vector<uint32_t*> hdRasterQueueVec;

		std::vector<DeviceVector<TileBinProposalCUDA>> hdCoverQueue;
		std::vector<TileBinProposalCUDA*> hdCoverQueueVec;

		TileBinProposalCUDA** dCoverQueue2;

		TileBinProposalCUDA* dCoverQueue;
		uint32_t* dCoverQueueCounter;
		uint32_t* dShadingQueue;
		uint32_t* dRasterQueueCounter;
		uint32_t* dAssembledTrianglesCounter;

		AssembledTriangleProposal** dAssembledTriangles;
		uint32_t** dRasterQueue;
		VaryingStore** dVaryingBuffer;
		TileRasterDeviceConstants* dDeviceConstants;

		AssembledTriangleProposalCUDA* dAssembledTriangles2;
		uint32_t* dAssembledTrianglesCounter2;
	};
}