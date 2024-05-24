#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterConstantsCuda.h"
namespace Ifrit::Engine::TileRaster::CUDA {
	void TileRasterRendererCuda::init() {
		context = std::make_unique<TileRasterContextCuda>();
		deviceContext = std::make_unique<TileRasterDeviceContext>();

		deviceContext->dCoverQueueCounter = (uint32_t*)Invocation::deviceMalloc(sizeof(uint32_t) * CU_TILE_SIZE * CU_TILE_SIZE);
		deviceContext->dShadingQueue = (uint32_t*)Invocation::deviceMalloc(sizeof(uint32_t));
		deviceContext->dRasterQueueCounter = (uint32_t*)Invocation::deviceMalloc(sizeof(uint32_t) * CU_TILE_SIZE * CU_TILE_SIZE);
		deviceContext->dAssembledTrianglesCounter2 = (uint32_t*)Invocation::deviceMalloc(sizeof(uint32_t));

		int totlTriangle = CU_SINGLE_TIME_TRIANGLE * 9;
		cudaMalloc(&deviceContext->dAssembledTriangles2, totlTriangle * sizeof(AssembledTriangleProposalCUDA));


		int totalTiles = CU_TILE_SIZE * CU_TILE_SIZE;
		int totalTriangles = CU_SINGLE_TIME_TRIANGLE;
		int maxProposals = CU_SINGLE_TIME_TRIANGLE;

		deviceContext->hdRasterQueueVec.resize(totalTiles);
		if (deviceContext->hdRasterQueue.size() < totalTiles) {
			deviceContext->hdRasterQueue.resize(totalTiles);
		}
		for (int i = 0; i < totalTiles; i++) {
			if (deviceContext->hdRasterQueue[i].size() < maxProposals) {
				deviceContext->hdRasterQueue[i].resize(maxProposals);
			}
			deviceContext->hdRasterQueueVec[i] = deviceContext->hdRasterQueue[i].data();
		}
		cudaMalloc(&deviceContext->dRasterQueue, totalTiles * sizeof(uint32_t*));
		cudaMemcpy(deviceContext->dRasterQueue, deviceContext->hdRasterQueueVec.data(), totalTiles * sizeof(uint32_t*), cudaMemcpyHostToDevice);


		deviceContext->hdCoverQueueVec.resize(totalTiles);
		if (deviceContext->hdCoverQueue.size() < totalTiles) {
			deviceContext->hdCoverQueue.resize(totalTiles);
		}
		for (int i = 0; i < totalTiles; i++) {
			if (deviceContext->hdCoverQueue[i].size() < maxProposals) {
				deviceContext->hdCoverQueue[i].resize(maxProposals);
			}
			deviceContext->hdCoverQueueVec[i] = deviceContext->hdCoverQueue[i].data();
		}
		cudaMalloc(&deviceContext->dCoverQueue2, totalTiles * sizeof(TileBinProposal*));
		cudaMemcpy(deviceContext->dCoverQueue2, deviceContext->hdCoverQueueVec.data(), totalTiles * sizeof(TileBinProposal*), cudaMemcpyHostToDevice);

		deviceContext->dDeviceConstants = (TileRasterDeviceConstants*)Invocation::deviceMalloc(sizeof(TileRasterDeviceConstants));
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
	
		needVaryingUpdate = true;
	
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
	void TileRasterRendererCuda::render() {
		initCuda();
		updateVaryingBuffer();

		ifloat4* colorBuffer = (ifloat4*)context->frameBuffer->getColorAttachment(0)->getData();
		TileRasterDeviceConstants hostConstants;
		hostConstants.attributeCount = context->vertexBuffer->getAttributeCount();
		hostConstants.counterClockwise = false;
		hostConstants.frameBufferHeight = context->frameBuffer->getHeight();
		hostConstants.frameBufferWidth = context->frameBuffer->getWidth();
		hostConstants.totalIndexCount = context->indexBuffer->size();
		hostConstants.varyingCount = context->varyingDescriptor->getVaryingCounts();
		hostConstants.vertexCount = context->vertexBuffer->getVertexCount();
		hostConstants.vertexStride = 3;

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
			&hostConstants,
			this->deviceContext.get(),
			this->doubleBuffer,
			deviceHostColorBuffers[1-curBuffer].data()
		);
		currentBuffer = 1 - curBuffer;
	}
}