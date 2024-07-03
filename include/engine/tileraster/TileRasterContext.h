#pragma once

#include "core/data/Image.h"
#include "engine/base/FrameBuffer.h"
#include "engine/base/Renderer.h"
#include "engine/base/VertexBuffer.h"
#include "engine/base/Shaders.h"
#include "engine/base/VertexShaderResult.h"
#include "engine/base/VaryingDescriptor.h"
#include "engine/tileraster/TileRasterCommon.h"

namespace Ifrit::Engine::TileRaster {
	class TileRasterContext {
	public:
		// Non-owning Bindings
		FrameBuffer* frameBuffer;
		const VertexBuffer* vertexBuffer;
		const std::vector<int>* indexBuffer;
		VertexShader* vertexShader;
		VaryingDescriptor* varyingDescriptor;
		FragmentShader* fragmentShader;

		// Resources
		std::unique_ptr<VertexShaderResult> vertexShaderResult;
		std::vector<std::vector<std::vector<TileBinProposal>>> rasterizerQueue;
		std::vector<std::vector<std::vector<TileBinProposal>>> coverQueue;

		// Config
		constexpr static int numThreads = 16;
		constexpr static int vertexStride = 3;
		constexpr static int tileBlocksX = 64;
		constexpr static int subtileBlocksX = 4;

		TileRasterFrontFace frontface = TileRasterFrontFace::CLOCKWISE;

		// Geometry
		std::vector<float> primitiveMinZ;
		std::vector<PrimitiveEdgeCoefs> primitiveEdgeCoefs;

		std::vector<std::vector<AssembledTriangleProposal>> assembledTriangles;

		// Profiling
		std::vector<uint32_t> workerIdleTime;
	};
}