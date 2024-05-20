#pragma once

#include "core/definition/CoreDefs.h"
#include "core/cuda/CudaUtils.cuh"
#include "engine/base/TypeDescriptor.h"

#include "engine/base/VertexShader.h"
#include "engine/base/FragmentShader.h"
#include "engine/tileraster/TileRasterCommon.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"

namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	constexpr float CU_EPS = 1e-7f;
}

namespace Ifrit::Engine::TileRaster::CUDA::Invocation {
	void testingKernelWrapper();

	void invokeCudaRendering(
		char* dVertexBuffer,
		TypeDescriptorEnum* dVertexTypeDescriptor,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		int* dIndexBuffer,
		VertexShader* dVertexShader,
		FragmentShader* dFragmentShader,
		ifloat4** hColorBuffer,
		float* dDepthBuffer,
		ifloat4*  dPositionBuffer,
		TileRasterDeviceConstants* deviceConstants,
		TileRasterDeviceContext* deviceContext
	);

	template<class T>T* copyShaderToDevice(T* x);

	int* getIndexBufferDeviceAddr(const int* hIndexBuffer, uint32_t indexBufferSize,int* dOldIndexBuffer);
	char* getVertexBufferDeviceAddr(const char* hVertexBuffer, uint32_t bufferSize, char* dOldBuffer);
	TypeDescriptorEnum* getTypeDescriptorDeviceAddr(const TypeDescriptorEnum* hBuffer, uint32_t bufferSize, TypeDescriptorEnum* dOldBuffer);
	float* getDepthBufferDeviceAddr( uint32_t bufferSize, float* dOldBuffer);
	ifloat4* getPositionBufferDeviceAddr(uint32_t bufferSize, ifloat4* dOldBuffer);

}