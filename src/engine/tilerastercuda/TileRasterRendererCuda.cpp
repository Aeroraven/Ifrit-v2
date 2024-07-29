#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterConstantsCuda.h"
#include "engine/tilerastercuda/TileRasterImageOpInvocationsCuda.cuh"
namespace Ifrit::Engine::TileRaster::CUDA {
	void TileRasterRendererCuda::init() {
		context = std::make_unique<TileRasterContextCuda>();
		deviceContext = std::make_unique<TileRasterDeviceContext>();

		deviceContext->dShadingQueue = (uint32_t*)Invocation::deviceMalloc(sizeof(uint32_t));
		deviceContext->dDeviceConstants = (TileRasterDeviceConstants*)Invocation::deviceMalloc(sizeof(TileRasterDeviceConstants));
		Invocation::initCudaRendering();
		context->geometryShader = nullptr;
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
		this->deviceIndexBuffer = Invocation::getIndexBufferDeviceAddr(indexBuffer.data(), indexBuffer.size(), this->deviceIndexBuffer);
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
		auto vxcount = context->vertexBuffer->getVertexCount();
		cudaMalloc(&deviceContext->dVaryingBufferM2, vcount * vxcount * sizeof(VaryingStore));
	}
	void TileRasterRendererCuda::bindFragmentShader(FragmentShader* fragmentShader) {
		context->fragmentShader = fragmentShader;
		needFragmentShaderUpdate = true;
	}
	void TileRasterRendererCuda::bindGeometryShader(GeometryShader* geometryShader) {
		context->geometryShader = geometryShader;
	}
	void TileRasterRendererCuda::createTextureRaw(int slotId, const IfritImageCreateInfo& createInfo, float* data) {
		Invocation::createTexture(slotId, createInfo, data);
		needFragmentShaderUpdate = true;
	}
	void TileRasterRendererCuda::createSampler(int slotId, const IfritSamplerT& samplerState) {
		Invocation::createSampler(slotId, samplerState);
		needFragmentShaderUpdate = true;
	}
	void TileRasterRendererCuda::setRasterizerPolygonMode(IfritPolygonMode mode) {
		this->polygonMode = mode;
	}
	void TileRasterRendererCuda::setBlendFunc(IfritColorAttachmentBlendState state) {
		Invocation::setBlendFunc(state);
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
	void TileRasterRendererCuda::generateMipmap(int slotId, IfritFilter filter) {
		Invocation::invokeMipmapGeneration(slotId, filter);
	}
	void TileRasterRendererCuda::blitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter) {
		Invocation::invokeBlitImage(srcSlotId, dstSlotId, region, filter);
	}
	void TileRasterRendererCuda::render() {
		initCuda();
		updateVaryingBuffer();

		if (needFragmentShaderUpdate) {
			Invocation::invokeFragmentShaderUpdate(context->fragmentShader);
			needFragmentShaderUpdate = false;
		}

		ifloat4* colorBuffer = (ifloat4*)context->frameBuffer->getColorAttachment(0)->getData();

		int totalIndices = context->indexBuffer->size();
		int curBuffer = currentBuffer;

		Invocation::RenderingInvocationArgumentSet args;
		args.dVertexBuffer = deviceVertexBuffer;
		args.dVertexTypeDescriptor = deviceVertexTypeDescriptor;
		args.dIndexBuffer = deviceIndexBuffer;
		args.dVertexShader = context->vertexShader;
		args.dFragmentShader = context->fragmentShader;
		args.dGeometryShader = context->geometryShader;
		args.dColorBuffer = deviceColorBuffer[curBuffer];
		args.dHostColorBuffer = deviceHostColorBuffers[curBuffer].data();
		args.hColorBuffer = hostColorBuffers.data();
		args.dHostColorBufferSize = deviceHostColorBuffers[curBuffer].size();
		args.dDepthBuffer = deviceDepthBuffer;
		args.dPositionBuffer = devicePosBuffer;
		args.deviceContext = this->deviceContext.get();
		args.totalIndices = totalIndices;
		args.doubleBuffering = this->doubleBuffer;
		args.dLastColorBuffer = deviceHostColorBuffers[1 - curBuffer].data();
		args.polygonMode = polygonMode;
		
		Invocation::invokeCudaRendering(args);
		currentBuffer = 1 - curBuffer;
	}
}