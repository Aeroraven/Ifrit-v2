#pragma once
#include "core/definition/CoreExports.h"

#if IFRIT_USE_CUDA
#include "engine/base/VaryingDescriptor.h"
#include "engine/tilerastercuda/TileRasterContextCuda.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "engine/base/Constants.h"

namespace Ifrit::Engine::TileRaster::CUDA {
	using namespace Ifrit::Engine;

	class TileRasterRendererCuda :public std::enable_shared_from_this<TileRasterRendererCuda> {
	private:
		std::unique_ptr<TileRasterContextCuda> context;
		std::unique_ptr<TileRasterDeviceContext> deviceContext;
		bool needVaryingUpdate = true;
		bool needFragmentShaderUpdate = true;
		bool initCudaContext = false;
		// Depth Test
		IfritCompareOp ctxDepthFunc = IF_COMPARE_OP_LESS;
		bool ctxDepthTestEnable = true;
		std::vector<ifloat4> ctxClearColors = { { 0.0f,0.0f,0.0f,0.0f } };
		float ctxClearDepth = 1.0f;

		// Device Addrs
		int* deviceIndexBuffer = nullptr;
		char* deviceVertexBuffer = nullptr;
		TypeDescriptorEnum* deviceVertexTypeDescriptor = nullptr;
		TypeDescriptorEnum* deviceVaryingTypeDescriptor = nullptr;
		float* deviceDepthBuffer = nullptr;
		ifloat4* devicePosBuffer = nullptr;
		int* deviceShadingLockBuffer = nullptr;

		std::vector<ifloat4*> deviceHostColorBuffers[2];
		ifloat4** deviceColorBuffer[2] = { nullptr,nullptr };
		std::vector<ifloat4*> hostColorBuffers{};

		bool doubleBuffer = false;
		int currentBuffer = 0;

		// Render confs
		IfritPolygonMode polygonMode = IF_POLYGON_MODE_FILL;

	private:
		void updateVaryingBuffer();

	public:
		void init();
		void initCuda();
		void bindFrameBuffer(FrameBuffer& frameBuffer, bool useDoubleBuffer = true);
		void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		void bindIndexBuffer(const std::vector<int>& indexBuffer);
		void bindVertexShader(VertexShader* vertexShader, VaryingDescriptor& varyingDescriptor);
		void bindFragmentShader(FragmentShader* fragmentShader);
		void bindGeometryShader(GeometryShader* geometryShader);

		void createTexture(int slotId, const IfritImageCreateInfo& createInfo);
		void createSampler(int slotId, const IfritSamplerT& samplerState);
		void generateMipmap(int slotId, IfritFilter filter);
		void blitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter);
		void copyHostBufferToImage(void* srcBuffer, int dstSlot, const std::vector<IfritBufferImageCopy>& regions);

		void setRasterizerPolygonMode(IfritPolygonMode mode);
		void setBlendFunc(IfritColorAttachmentBlendState state);
		void setDepthFunc(IfritCompareOp depthFunc);
		void setDepthTestEnable(bool option);
		void setCullMode(IfritCullMode cullMode);
		void setClearValues(const std::vector<ifloat4>& clearColors, float clearDepth);

		void clear();
		void render();
	};
}
#endif