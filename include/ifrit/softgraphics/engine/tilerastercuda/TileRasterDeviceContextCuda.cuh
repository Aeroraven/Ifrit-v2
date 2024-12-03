
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


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