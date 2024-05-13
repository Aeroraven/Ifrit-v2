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
		float4 e1, e2, e3; //Edge Coefs
		float3 f1, f2, f3; //Interpolate Bases
		float iw1, iw2, iw3; //Inversed W
		rect2Df bbox;
		int originalPrimitive;
	};

	struct AssembledTriangleProposalReference {
		uint32_t workerId;
		int primId;
	};

	struct TileBinProposal {
		int2 tile;
		TileRasterLevel level;
		AssembledTriangleProposalReference clippedTriangle;
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
		int numThreads = 16;
		int vertexStride = 3;
		int tileBlocksX = 64;
		int subtileBlocksX = 4;

		TileRasterFrontFace frontface = TileRasterFrontFace::CLOCKWISE;

		// Geometry
		std::vector<float> primitiveMinZ;
		std::vector<PrimitiveEdgeCoefs> primitiveEdgeCoefs;

		std::vector<std::vector<AssembledTriangleProposal>> assembledTriangles;

		// Profiling
		std::vector<uint32_t> workerIdleTime;
	};
}