#pragma once

#include "core/definition/CoreDefs.h"
#include "core/cuda/CudaUtils.cuh"
#include "engine/base/TypeDescriptor.h"

#include "engine/base/VertexShader.h"
#include "engine/base/FragmentShader.h"
#include "engine/tileraster/TileRasterCommon.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"

namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	// Constants
	constexpr float CU_EPS = 1e-7f;
}

namespace Ifrit::Engine::TileRaster::CUDA::Invocation {
	void testingKernelWrapper();

	void invokeCudaRendering(
		char* hVertexBuffer,
		uint32_t hVertexBufferSize,
		TypeDescriptorEnum* hVertexTypeDescriptor,
		TypeDescriptorEnum* hVaryingTypeDescriptor,
		int* hIndexBuffer,
		VertexShader* dVertexShader,
		FragmentShader* dFragmentShader,
		ifloat4** hColorBuffer,
		float* hDepthBuffer,
		TileRasterDeviceConstants* deviceConstants,
		TileRasterDeviceContext* deviceContext
	);

	template<class T>T* copyShaderToDevice(T* x);

}