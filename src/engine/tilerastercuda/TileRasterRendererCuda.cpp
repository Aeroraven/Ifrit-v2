#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"

namespace Ifrit::Engine::TileRaster::CUDA {
	void TileRasterRendererCuda::init() {
		context = std::make_unique<TileRasterContextCuda>();
		deviceContext = std::make_unique<TileRasterDeviceContext>();
	}
	void TileRasterRendererCuda::bindFrameBuffer(FrameBuffer& frameBuffer) {
		context->frameBuffer = &frameBuffer;
	}	
	void TileRasterRendererCuda::bindVertexBuffer(const VertexBuffer& vertexBuffer) {
		context->vertexBuffer = &vertexBuffer;
	}
	void TileRasterRendererCuda::bindIndexBuffer(const std::vector<int>& indexBuffer) {
		context->indexBuffer = &indexBuffer;
	}
	void TileRasterRendererCuda::bindVertexShader(VertexShader& vertexShader, VaryingDescriptor& varyingDescriptor) {
		context->vertexShader = &vertexShader;
		context->varyingDescriptor = &varyingDescriptor;
	}
	void TileRasterRendererCuda::bindFragmentShader(FragmentShader& fragmentShader) {
		context->fragmentShader = &fragmentShader;
	}
	void TileRasterRendererCuda::clear() {
		context->frameBuffer->getDepthAttachment()->clearImage(255.0);
		context->frameBuffer->getColorAttachment(0)->clearImageZero();
	}

	void TileRasterRendererCuda::render() {
		char* hVertexBuffer = context->vertexBuffer->getBufferUnsafe();
		uint32_t hVertexBufferSize = context->vertexBuffer->getBufferSize();
		std::vector<TypeDescriptorEnum> hVertexBufferLayout;
		std::vector<TypeDescriptorEnum> hVaryingBufferLayout;
		for (int i = 0; i < context->vertexBuffer->getAttributeCount(); i++) {
			hVertexBufferLayout.push_back(context->vertexBuffer->getAttributeDescriptor(i).type);
		}
		for (int i = 0; i < context->varyingDescriptor->getVaryingCounts(); i++) {
			hVaryingBufferLayout.push_back(context->varyingDescriptor->getVaryingDescriptor(i).type);
		}
		ifloat4* colorBuffer = (ifloat4*)context->frameBuffer->getColorAttachment(0)->getData();

		TileRasterDeviceConstants hostConstants;
		hostConstants.attributeCount = context->vertexBuffer->getAttributeCount();
		hostConstants.counterClockwise = false;
		hostConstants.frameBufferHeight = context->frameBuffer->getHeight();
		hostConstants.frameBufferWidth = context->frameBuffer->getWidth();
		hostConstants.startingIndexId = 0;
		hostConstants.totalIndexCount = context->indexBuffer->size();
		hostConstants.varyingCount = context->varyingDescriptor->getVaryingCounts();
		hostConstants.vertexCount = context->vertexBuffer->getVertexCount();
		hostConstants.vertexStride = 3;

		Invocation::invokeCudaRendering(
			hVertexBuffer,
			hVertexBufferSize,
			hVertexBufferLayout.data(),
			hVaryingBufferLayout.data(),
			(int*)context->indexBuffer->data(),
			context->vertexShader,
			context->fragmentShader,
			&colorBuffer,
			(float*)context->frameBuffer->getDepthAttachment()->getData(),
			&hostConstants,
			this->deviceContext.get()
		);
	}
}