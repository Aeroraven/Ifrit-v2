#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"

namespace Ifrit::Engine::TileRaster::CUDA {
	void TileRasterRendererCuda::init() {
		context = std::make_unique<TileRasterContextCuda>();
		deviceContext = std::make_unique<TileRasterDeviceContext>();
	}
	void TileRasterRendererCuda::bindFrameBuffer(FrameBuffer& frameBuffer) {
		context->frameBuffer = &frameBuffer;
		auto pixelCount = frameBuffer.getWidth() * frameBuffer.getHeight();
		this->deviceDepthBuffer = Invocation::getDepthBufferDeviceAddr(pixelCount, this->deviceDepthBuffer);
	}	
	void TileRasterRendererCuda::bindVertexBuffer(const VertexBuffer& vertexBuffer) {
		context->vertexBuffer = &vertexBuffer;
		char* hVertexBuffer = context->vertexBuffer->getBufferUnsafe();
		uint32_t hVertexBufferSize = context->vertexBuffer->getBufferSize();
		this->deviceVertexBuffer = Invocation::getVertexBufferDeviceAddr(hVertexBuffer, hVertexBufferSize, this->deviceVertexBuffer);
		this->devicePosBuffer = Invocation::getPositionBufferDeviceAddr(hVertexBufferSize, this->devicePosBuffer);

		std::vector<TypeDescriptorEnum> hVertexBufferLayout;
		for (int i = 0; i < context->vertexBuffer->getAttributeCount(); i++) {
			hVertexBufferLayout.push_back(context->vertexBuffer->getAttributeDescriptor(i).type);
		}
		this->deviceVertexTypeDescriptor = Invocation::getTypeDescriptorDeviceAddr(hVertexBufferLayout.data(), hVertexBufferLayout.size(), this->deviceVertexTypeDescriptor);
	}
	void TileRasterRendererCuda::bindIndexBuffer(const std::vector<int>& indexBuffer) {
		context->indexBuffer = &indexBuffer;
		this->deviceIndexBuffer = Invocation::getIndexBufferDeviceAddr(indexBuffer.data(), indexBuffer.size(),this->deviceIndexBuffer);
	}

	void TileRasterRendererCuda::bindVertexShader(VertexShader* vertexShader, VaryingDescriptor& varyingDescriptor) {
		context->vertexShader = vertexShader;
		context->varyingDescriptor = &varyingDescriptor;
		std::vector<TypeDescriptorEnum> hVaryingBufferLayout;
		for (int i = 0; i < context->varyingDescriptor->getVaryingCounts(); i++) {
			hVaryingBufferLayout.push_back(context->varyingDescriptor->getVaryingDescriptor(i).type);
		}
		this->deviceVaryingTypeDescriptor = Invocation::getTypeDescriptorDeviceAddr(hVaryingBufferLayout.data(), hVaryingBufferLayout.size(), this->deviceVaryingTypeDescriptor);
	}

	void TileRasterRendererCuda::bindFragmentShader(FragmentShader* fragmentShader) {
		context->fragmentShader = fragmentShader;
	}
	void TileRasterRendererCuda::clear() {
		context->frameBuffer->getDepthAttachment()->clearImage(255.0);
		context->frameBuffer->getColorAttachment(0)->clearImageZero();
	}

	void TileRasterRendererCuda::render() {

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
			deviceVertexBuffer,
			deviceVertexTypeDescriptor,
			deviceVaryingTypeDescriptor,
			deviceIndexBuffer,
			context->vertexShader,
			context->fragmentShader,
			&colorBuffer,
			deviceDepthBuffer,
			devicePosBuffer,
			&hostConstants,
			this->deviceContext.get()
		);
	}
}