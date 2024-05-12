#pragma once

#include "core/data/Image.h"
#include "engine/base/FrameBuffer.h"
#include "engine/base/Renderer.h"
#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShader.h"
#include "engine/base/FragmentShader.h"
#include "engine/base/VertexShaderResult.h"

namespace Ifrit::Engine::TileRaster {
	enum class TileRasterHomogeneousClipping {
		DISABLE,
		SIMPLE_DISCARD,
		HOMOGENEOUS_CLIPPING
	};

	enum class TileRasterFrontFace {
		CLOCKWISE,
		COUNTER_CLOCKWISE
	};

	enum class TileRasterLevel {
		TILE,
		BLOCK,
		PIXEL,
		PIXEL_PACK2X2,	//SIMD128
		PIXEL_PACK4X2,	//SIMD256
		PIXEL_PACK4X4,	//SIMD512
	};

	struct AssembledTriangleProposal {
		float4 v1, v2, v3;
		float4 b1, b2, b3;
		int originalPrimitive;
	};

	struct TileBinProposal {
		int primitiveId;
		rect2Df bbox;
		int2 tile;
		bool allAccept;
		TileRasterLevel level;
		AssembledTriangleProposal clippedTriangle;
	};

	struct PrimitiveEdgeCoefs {
		float3 coef[3];
	};

	class TileRasterContext {
	public:
		// Non-owning Bindings
		FrameBuffer* frameBuffer;
		const VertexBuffer* vertexBuffer;
		const std::vector<int>* indexBuffer;
		VertexShader* vertexShader;
		FragmentShader* fragmentShader;

		// Resources
		std::unique_ptr<VertexShaderResult> vertexShaderResult;
		std::vector<std::vector<std::vector<TileBinProposal>>> rasterizerQueue;
		std::vector<std::vector<std::vector<TileBinProposal>>> coverQueue;

		// Config
		int numThreads = 10;
		int vertexStride = 3;
		int tileBlocksX = 64;
		int subtileBlocksX = 4;

		TileRasterFrontFace frontface = TileRasterFrontFace::CLOCKWISE;

		// Geometry
		std::vector<float> primitiveMinZ;
		std::vector<PrimitiveEdgeCoefs> primitiveEdgeCoefs;

		// Profiling
		std::vector<uint32_t> workerIdleTile;
	};
}