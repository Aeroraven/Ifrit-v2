#pragma once
#include "core/definition/CoreExports.h"

#ifdef IFRIT_FEATURE_CUDA
#include "engine/tilerastercuda/TileRasterContextCuda.h"

namespace Ifrit::Engine::TileRaster::CUDA {
	using namespace Ifrit::Engine;

	namespace Invocation {
		IFRIT_KERNEL void vertexProcessingKernel(
			VertexShader* vertexShader,
			char** dVertexBuffer,
			TypeDescriptorEnum* dVertexTypeDescriptor,
			char** dVaryingBuffer,
			TypeDescriptorEnum* dVaryingTypeDescriptor,
			float4* dPosBuffer,
			TileRasterDeviceConstants* deviceConstants
		);

		IFRIT_KERNEL void geometryProcessingKernel(
			float4* dPosBuffer,
			int* dIndexBuffer,
			AssembledTriangleProposal** dAssembledTriangles,
			uint32_t* dAssembledTriangleCount,
			TileBinProposal*** dRasterQueue,
			uint32_t** dRasterQueueCount,
			TileBinProposal*** dCoverQueue,
			uint32_t** dCoverQueueCount,
			TileRasterDeviceConstants* deviceConstants
		);

		IFRIT_KERNEL void tilingRasterizationKernel(
			AssembledTriangleProposal** dAssembledTriangles,
			uint32_t* dAssembledTriangleCount,
			TileBinProposal*** dRasterQueue,
			uint32_t** dRasterQueueCount,
			TileBinProposal*** dCoverQueue,
			TileRasterDeviceConstants* deviceConstants
		);

		IFRIT_KERNEL void fragmentShadingKernel(
			AssembledTriangleProposal** dAssembledTriangles,
			uint32_t* dAssembledTriangleCount,
			TileBinProposal*** dCoverQueue,
			float** dColorBuffer,
			float* dDepthBuffer,
			TileRasterDeviceConstants* deviceConstants
		);
	}

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