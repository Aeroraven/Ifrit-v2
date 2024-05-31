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

		uint32_t* dShadingQueue;

		VaryingStore** dVaryingBuffer;
		TileRasterDeviceConstants* dDeviceConstants;
	};
}