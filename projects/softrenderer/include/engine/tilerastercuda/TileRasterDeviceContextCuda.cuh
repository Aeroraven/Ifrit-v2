#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "core/definition/CoreDefs.h"
#include "core/cuda/CudaUtils.cuh"
#include "engine/tileraster/TileRasterCommon.h"
#include "engine/base/VaryingStore.h"
#include <vector>

namespace Ifrit::Engine::SoftRenderer::TileRaster::CUDA {
	using Ifrit::Engine::SoftRenderer::Core::CUDA::DeviceVector;
	using namespace Ifrit::Engine::SoftRenderer::TileRaster;
	using namespace Ifrit::Engine::SoftRenderer;

	struct TileRasterDeviceContext {
		//std::vector<DeviceVector<float4>> hdVaryingBuffer;
		//std::vector<float4*> hdVaryingBufferVec;
		uint32_t* dShadingQueue;
		//float4** dVaryingBuffer;
		TileRasterDeviceConstants* dDeviceConstants;
		float4* dVaryingBufferM2;
	};
}
#endif