#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterConstantsCuda.h"
namespace Ifrit::Engine::TileRaster::CUDA {
	void TileRasterRendererCuda::init() {
		context = std::make_unique<TileRasterContextCuda>();
		deviceContext = std::make_unique<TileRasterDeviceContext>();

		deviceContext->dShadingQueue = (uint32_t*)Invocation::deviceMalloc(sizeof(uint32_t));

		int totlTriangle = CU_SINGLE_TIME_TRIANGLE * 2 ;
		int totalTiles = CU_TILE_SIZE * CU_TILE_SIZE;
		int totalTriangles = CU_SINGLE_TIME_TRIANGLE;
		int maxProposals = CU_SINGLE_TIME_TRIANGLE;

		deviceContext->dDeviceConstants = (TileRasterDeviceConstants*)Invocation::deviceMalloc(sizeof(TileRasterDeviceConstants));
		Invocation::initCudaRendering();
	}
	void TileRasterRendererCuda::bindFrameBuffer(FrameBuffer& frameBuffer, bool useDoubleBuffer) {
		context->frameBuffer = &frameBuffer;
		auto pixelCount = frameBuffer.getWidth() * frameBuffer.getHeight();
		this->deviceDepthBuffer = Invocation::getDepthBufferDeviceAddr(pixelCount, this->deviceDepthBuffer);

		std::vector<ifloat4*> hColorBuffer = { (ifloat4*)frameBuffer.getColorAttachment(0)->getData() };
	
		Invocation::getColorBufferDeviceAddr(hColorBuffer,
			this->deviceHostColorBuffers[0], this->deviceColorBuffer[0], pixelCount, this->deviceHostColorBuffers[0], this->deviceColorBuffer[0]);
		Invocation::getColorBufferDeviceAddr(hColorBuffer,
			this->deviceHostColorBuffers[1], this->deviceColorBuffer[1], pixelCount, this->deviceHostColorBuffers[1], this->deviceColorBuffer[1]);
		this->doubleBuffer = useDoubleBuffer;
		this->hostColorBuffers = hColorBuffer;
		Invocation::updateFrameBufferConstants(frameBuffer.getWidth(), frameBuffer.getHeight());
	}	
	void TileRasterRendererCuda::bindVertexBuffer(const VertexBuffer& vertexBuffer) {
		needVaryingUpdate = true;
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

		Invocation::updateVertexLayout(hVertexBufferLayout.data(), hVertexBufferLayout.size());
		Invocation::updateVertexCount(vertexBuffer.getVertexCount());
		Invocation::updateAttributes(vertexBuffer.getAttributeCount());
	}
	void TileRasterRendererCuda::bindIndexBuffer(const std::vector<int>& indexBuffer) {
		context->indexBuffer = &indexBuffer;
		if constexpr (CU_OPT_ALIGNED_INDEX_BUFFER) {
			std::vector<int> indexBufferInt4;
			for (int i = 0; i < indexBuffer.size(); i++) {
				indexBufferInt4.push_back(indexBuffer[i]);
				if (i % 3 == 2)indexBufferInt4.push_back(0);
			}
			this->deviceIndexBuffer = Invocation::getIndexBufferDeviceAddr(indexBufferInt4.data(), indexBufferInt4.size(), this->deviceIndexBuffer);
		}
		else {
			this->deviceIndexBuffer = Invocation::getIndexBufferDeviceAddr(indexBuffer.data(), indexBuffer.size(), this->deviceIndexBuffer);
		}
		
	}

	void TileRasterRendererCuda::bindVertexShader(VertexShader* vertexShader, VaryingDescriptor& varyingDescriptor) {
		context->vertexShader = vertexShader;
		context->varyingDescriptor = &varyingDescriptor;
		std::vector<TypeDescriptorEnum> hVaryingBufferLayout;
		for (int i = 0; i < context->varyingDescriptor->getVaryingCounts(); i++) {
			hVaryingBufferLayout.push_back(context->varyingDescriptor->getVaryingDescriptor(i).type);
		}
		this->deviceVaryingTypeDescriptor = Invocation::getTypeDescriptorDeviceAddr(hVaryingBufferLayout.data(), hVaryingBufferLayout.size(), this->deviceVaryingTypeDescriptor);
	
		needVaryingUpdate = true;
		Invocation::updateVarying(context->varyingDescriptor->getVaryingCounts());
	}
	void TileRasterRendererCuda::updateVaryingBuffer() {
		if (!needVaryingUpdate)return;
		needVaryingUpdate = false;
		auto vcount = context->varyingDescriptor->getVaryingCounts();
		deviceContext->hdVaryingBufferVec.resize(vcount);
		if (deviceContext->hdVaryingBuffer.size() < vcount) {
			deviceContext->hdVaryingBuffer.resize(vcount);
		}
		for (int i = 0; i < vcount; i++) {
			if (deviceContext->hdVaryingBuffer[i].size() < context->vertexBuffer->getVertexCount()) {
				deviceContext->hdVaryingBuffer[i].resize(context->vertexBuffer->getVertexCount());
			}
			deviceContext->hdVaryingBufferVec[i] = deviceContext->hdVaryingBuffer[i].data();
		}

		cudaMalloc(&deviceContext->dVaryingBuffer, vcount * sizeof(VaryingStore*));
		cudaMemcpy(deviceContext->dVaryingBuffer, deviceContext->hdVaryingBufferVec.data(), vcount * sizeof(VaryingStore*), cudaMemcpyHostToDevice);

	}
	void TileRasterRendererCuda::bindFragmentShader(FragmentShader* fragmentShader) {
		context->fragmentShader = fragmentShader;
	}
	void TileRasterRendererCuda::clear() {
		context->frameBuffer->getDepthAttachment()->clearImage(255.0);
		context->frameBuffer->getColorAttachment(0)->clearImageZero();
	}
	void TileRasterRendererCuda::initCuda() {
		if (this->initCudaContext)return;
		this->initCudaContext = true;
		cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 8192);
	}
	void TileRasterRendererCuda::setAggressiveRatio(float ratio) {
		this->aggressiveRatio = ratio;
	}
	void TileRasterRendererCuda::render() {
		initCuda();
		updateVaryingBuffer();

		ifloat4* colorBuffer = (ifloat4*)context->frameBuffer->getColorAttachment(0)->getData();

		int totalIndices = context->indexBuffer->size();
		if constexpr (CU_OPT_ALIGNED_INDEX_BUFFER) {
			totalIndices = totalIndices / 3 * 4;
		}

		int curBuffer = currentBuffer;
		Invocation::invokeCudaRendering(
			deviceVertexBuffer,
			deviceVertexTypeDescriptor,
			deviceVaryingTypeDescriptor,
			deviceIndexBuffer,
			deviceShadingLockBuffer,
			context->vertexShader,
			context->fragmentShader,
			deviceColorBuffer[curBuffer],
			deviceHostColorBuffers[curBuffer].data(),
			hostColorBuffers.data(),
			deviceHostColorBuffers[curBuffer].size(),
			deviceDepthBuffer,
			devicePosBuffer,
			this->deviceContext.get(),
			totalIndices,
			this->doubleBuffer,
			deviceHostColorBuffers[1-curBuffer].data(),
			aggressiveRatio
		);
		currentBuffer = 1 - curBuffer;
	}
}