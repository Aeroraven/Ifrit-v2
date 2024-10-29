#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
#include "ifrit/softgraphics/core/cuda/CudaUtils.cuh"
#include "ifrit/softgraphics/engine/tileraster/TileRasterCommon.h"
#include "ifrit/softgraphics/engine/base/VaryingStore.h"
#include <vector>

namespace Ifrit::SoftRenderer::TileRaster::CUDA {
	using Ifrit::SoftRenderer::Core::CUDA::DeviceVector;
	using namespace Ifrit::SoftRenderer::TileRaster;
	using namespace Ifrit::SoftRenderer;

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