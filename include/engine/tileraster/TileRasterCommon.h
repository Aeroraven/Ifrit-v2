#pragma once
#include "core/definition/CoreDefs.h"
#include "core/definition/CoreTypes.h"

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
		ifloat4 v1, v2, v3;
		ifloat4 b1, b2, b3;
		ifloat3 e1, e2, e3; //Edge Coefs
		ifloat4 f1, f2, f3; //Interpolate Bases
		int originalPrimitive;
	};

	struct AssembledTriangleProposalReference {
		uint32_t workerId;
		int primId;
	};

	struct TileBinProposal {
		iint2 tile;
		TileRasterLevel level;
		AssembledTriangleProposalReference clippedTriangle;
	};

	struct PrimitiveEdgeCoefs {
		ifloat3 coef[3];
	};

	class TileRasterDeviceConstants {
	public:
		int vertexStride = 3;
		int tileBlocksX = 64;
		int subtileBlocksX = 2;
		int vertexProcessingThreads = 128;
		int geometryProcessingThreads = 128;
		int tilingRasterizationThreads = 16;
		int fragmentProcessingThreads = 8;
		int vertexCount;
		int attributeCount;
		int varyingCount;
		int indexCount;
		int frameBufferWidth;
		int frameBufferHeight;

		bool counterClockwise = false;
		int startingIndexId;
		int totalIndexCount;
	};

	struct TileRasterClipVertex {
		ifloat4 barycenter;
		ifloat4 pos;
	};
}