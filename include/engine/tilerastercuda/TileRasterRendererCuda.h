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
	public:
		void init();
		void bindFrameBuffer(FrameBuffer& frameBuffer);
		void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		void bindIndexBuffer(const std::vector<int>& indexBuffer);
		void bindVertexShader(VertexShader& vertexShader, VaryingDescriptor& varyingDescriptor);
		void bindFragmentShader(FragmentShader& fragmentShader);

		void clear();
		void render();
	};
}
#endif