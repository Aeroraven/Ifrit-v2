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
		enum TileRasterRendererCudaVertexPipelineType {
			IFINTERNAL_CU_VERTEX_PIPELINE_UNDEFINED = 0,
			IFINTERNAL_CU_VERTEX_PIPELINE_CONVENTIONAL = 1,
			IFINTERNAL_CU_VERTEX_PIPELINE_MESHSHADER = 2
		};

	private:
		void updateVaryingBuffer();
		void internalRender(TileRasterRendererCudaVertexPipelineType vertexPipeType);
		void initCuda();

	public:
		IFRIT_APIDECL void init();
		IFRIT_APIDECL void bindFrameBuffer(FrameBuffer& frameBuffer, bool useDoubleBuffer = true);
		IFRIT_APIDECL void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		IFRIT_APIDECL void bindIndexBuffer(const std::vector<int>& indexBuffer);
		IFRIT_APIDECL void bindVertexShader(VertexShader* vertexShader, VaryingDescriptor& varyingDescriptor);
		IFRIT_APIDECL void bindFragmentShader(FragmentShader* fragmentShader);
		IFRIT_APIDECL void bindGeometryShader(GeometryShader* geometryShader);
		IFRIT_APIDECL void bindMeshShader(MeshShader* meshShader, VaryingDescriptor& varyingDescriptor, iint3 localSize);
		IFRIT_APIDECL void bindTaskShader(TaskShader* taskShader, VaryingDescriptor& varyingDescriptor);

		IFRIT_APIDECL void createTexture(int slotId, const IfritImageCreateInfo& createInfo);
		IFRIT_APIDECL void createSampler(int slotId, const IfritSamplerT& samplerState);
		IFRIT_APIDECL void generateMipmap(int slotId, IfritFilter filter);
		IFRIT_APIDECL void blitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter);
		IFRIT_APIDECL void copyHostBufferToImage(void* srcBuffer, int dstSlot, const std::vector<IfritBufferImageCopy>& regions);

		IFRIT_APIDECL void createBuffer(int slotId, int bufSize);
		IFRIT_APIDECL void copyHostBufferToBuffer(const void* srcBuffer, int dstSlot, int size);

		IFRIT_APIDECL void setScissors(const std::vector<ifloat4>& scissors);
		IFRIT_APIDECL void setScissorTestEnable(bool option);

		IFRIT_APIDECL void setMsaaSamples(IfritSampleCountFlagBits msaaSamples);

		IFRIT_APIDECL void setRasterizerPolygonMode(IfritPolygonMode mode);
		IFRIT_APIDECL void setBlendFunc(IfritColorAttachmentBlendState state);
		IFRIT_APIDECL void setDepthFunc(IfritCompareOp depthFunc);
		IFRIT_APIDECL void setDepthTestEnable(bool option);
		IFRIT_APIDECL void setCullMode(IfritCullMode cullMode);
		IFRIT_APIDECL void setClearValues(const std::vector<ifloat4>& clearColors, float clearDepth);

		IFRIT_APIDECL void clear();
		IFRIT_APIDECL void drawElements();
		IFRIT_APIDECL void drawMeshTasks(int numWorkGroups, int firstWorkGroup);
	};
}
#endif