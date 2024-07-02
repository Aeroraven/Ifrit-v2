#pragma once
#include "core/definition/CoreExports.h"

#if IFRIT_USE_CUDA
#include "engine/base/VaryingDescriptor.h"
#include "engine/tilerastercuda/TileRasterContextCuda.h"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterInvocationCuda.cuh"

namespace Ifrit::Engine::TileRaster::CUDA {
	using namespace Ifrit::Engine;

	class TileRasterRendererCuda:public std::enable_shared_from_this<TileRasterRendererCuda> {
	private:
		std::unique_ptr<TileRasterContextCuda> context;
		std::unique_ptr<TileRasterDeviceContext> deviceContext;
		bool needVaryingUpdate = true;
		bool needFragmentShaderUpdate = true;
		bool initCudaContext = false;

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
		
		void createTextureRaw(int slotId, int height, int width, float* data);

		void clear();
		void render();
	};
}
#endif