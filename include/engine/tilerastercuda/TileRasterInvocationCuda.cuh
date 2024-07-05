#pragma once

#include "core/definition/CoreDefs.h"
#include "core/cuda/CudaUtils.cuh"
#include "engine/base/TypeDescriptor.h"

#include "engine/base/Shaders.h"
#include "engine/tileraster/TileRasterCommon.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/base/Constants.h"
#include "engine/base/Structures.h"

namespace Ifrit::Engine::TileRaster::CUDA::Invocation {
	

	struct RenderingInvocationArgumentSet {
		char* dVertexBuffer;
		TypeDescriptorEnum* dVertexTypeDescriptor;
		int* dIndexBuffer;
		VertexShader* dVertexShader;
		FragmentShader* dFragmentShader;
		GeometryShader* dGeometryShader;
		ifloat4** dColorBuffer;
		ifloat4** dHostColorBuffer;
		ifloat4** hColorBuffer;
		uint32_t dHostColorBufferSize;
		float* dDepthBuffer;
		ifloat4* dPositionBuffer;
		TileRasterDeviceContext* deviceContext;
		int totalIndices;
		bool doubleBuffering;
		ifloat4** dLastColorBuffer;
		IfritPolygonMode polygonMode = IF_POLYGON_MODE_FILL;
	};

	void invokeCudaRendering(const RenderingInvocationArgumentSet& args) IFRIT_AP_NOTHROW;

	void invokeFragmentShaderUpdate(FragmentShader* dFragmentShader) IFRIT_AP_NOTHROW;
	void updateFrameBufferConstants(uint32_t width, uint32_t height);
	void initCudaRendering();
	void updateVertexLayout(TypeDescriptorEnum* dVertexTypeDescriptor, int attrCounts);

	int* getIndexBufferDeviceAddr(const int* hIndexBuffer, uint32_t indexBufferSize,int* dOldIndexBuffer);
	char* getVertexBufferDeviceAddr(const char* hVertexBuffer, uint32_t bufferSize, char* dOldBuffer);
	TypeDescriptorEnum* getTypeDescriptorDeviceAddr(const TypeDescriptorEnum* hBuffer, uint32_t bufferSize, TypeDescriptorEnum* dOldBuffer);
	float* getDepthBufferDeviceAddr( uint32_t bufferSize, float* dOldBuffer);
	ifloat4* getPositionBufferDeviceAddr(uint32_t bufferSize, ifloat4* dOldBuffer);
	void getColorBufferDeviceAddr(const std::vector<ifloat4*>& hColorBuffer, std::vector<ifloat4*>& dhColorBuffer, ifloat4**& dColorBuffer, uint32_t bufferSize, std::vector<ifloat4*>& dhOldColorBuffer, ifloat4** dOldBuffer);
	void updateAttributes(uint32_t attributeCounts);
	void updateVarying(uint32_t varyingCounts);
	void updateVertexCount(uint32_t vertexCount);

	char* deviceMalloc(uint32_t size);
	void deviceFree(char* ptr);
	void createTexture(uint32_t texId, uint32_t texWid, uint32_t texHeight, float *data);
	void createSampler(uint32_t slotId, const IfritSamplerT& samplerState);
}