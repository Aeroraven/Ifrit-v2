#pragma once
#include "core/definition/CoreDefs.h"
#include "core/definition/CoreTypes.h"

#include "core/data/Image.h"
#include "engine/base/FrameBuffer.h"
#include "engine/base/Renderer.h"
#include "engine/base/VertexBuffer.h"
#include "engine/base/Shaders.h"
#include "engine/base/VertexShaderResult.h"
#include "engine/tileraster/TileRasterCommon.h"
#include "engine/base/VaryingDescriptor.h"

namespace Ifrit::Engine::TileRaster::CUDA {
	using namespace Ifrit::Engine::TileRaster;
	using namespace Ifrit::Engine;

	

	class TileRasterContextCuda {
	public:
		// Non-owning Bindings
		FrameBuffer* frameBuffer;
		const VertexBuffer* vertexBuffer;
		const std::vector<int>* indexBuffer;
		VertexShader* vertexShader;
		FragmentShader* fragmentShader;
		GeometryShader* geometryShader;
		VaryingDescriptor* varyingDescriptor;

		// Constants
		int vertexStride = 3;
		int tileBlocksX = 64;
		int subtileBlocksX = 4;
		int vertexProcessingThreads = 128;
		TileRasterDeviceConstants hostConstants;

		// CUDA Contexts
		TileRasterDeviceConstants* deviceConstants;

		char** dVertexBuffer = nullptr;
		int* dIndexBuffer = nullptr;
		char** dVaryingBuffer = nullptr;
		float** dColorBuffer = nullptr;
		float* dDepthBuffer = nullptr;

		TileBinProposal*** dRasterizerQueue = nullptr;
		TileBinProposal*** dCoverQueue = nullptr;
		AssembledTriangleProposal** dAssembledTriangles = nullptr;

		uint32_t** dRasterizerQueueSize = nullptr;
		uint32_t** dCoverQueueSize = nullptr;
		uint32_t* dAssembledTrianglesSize = nullptr;

		irect2Df* dTileBounds = nullptr;
	};
}