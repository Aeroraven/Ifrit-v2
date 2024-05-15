#pragma once
#include "core/definition/CoreExports.h"

#if IFRIT_USE_CUDA
#include "engine/tilerastercuda/TileRasterContextCuda.h"

namespace Ifrit::Engine::TileRaster::CUDA {
	using namespace Ifrit::Engine;

	

	class TileRasterRendererCuda {
	private:
		std::unique_ptr<TileRasterContextCuda> context;
	public:
		TileRasterRendererCuda();
		void bindFrameBuffer(FrameBuffer& frameBuffer);
		void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		void bindIndexBuffer(const std::vector<int>& indexBuffer);
		void deviceHostSync();
		void bindVertexShader(VertexShader& vertexShader);
		void bindFragmentShader(FragmentShader& fragmentShader);
		void intializeRenderContext();

		void deviceClear();
		void vertexProcessing();
		void geometryProcessing();
		void rasterization();
		void fragmentShading();
	};
}
#endif