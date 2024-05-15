#pragma once
#include "core/definition/CoreExports.h"
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
		float3 e1, e2, e3; //Edge Coefs
		float3 f1, f2, f3; //Interpolate Bases
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
}