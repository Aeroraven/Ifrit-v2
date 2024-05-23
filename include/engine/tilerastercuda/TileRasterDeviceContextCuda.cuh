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

		std::vector<DeviceVector<TileBinProposalCUDA>> hdRasterQueue;
		std::vector<TileBinProposalCUDA*> hdRasterQueueVec;

		std::vector<DeviceVector<TileBinProposalCUDA>> hdCoverQueue;
		std::vector<TileBinProposalCUDA*> hdCoverQueueVec;

		TileBinProposalCUDA** dCoverQueue2;

		TileBinProposalCUDA* dCoverQueue;
		uint32_t* dCoverQueueCounter;
		uint32_t* dShadingQueue;
		uint32_t* dRasterQueueCounter;
		uint32_t* dAssembledTrianglesCounter;

		AssembledTriangleProposal** dAssembledTriangles;
		TileBinProposalCUDA** dRasterQueue;
		VaryingStore** dVaryingBuffer;
		TileRasterDeviceConstants* dDeviceConstants;

		AssembledTriangleProposal* dAssembledTriangles2;
		uint32_t* dAssembledTrianglesCounter2;
	};
}