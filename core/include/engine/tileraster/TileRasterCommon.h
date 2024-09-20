#pragma once
#include "core/definition/CoreDefs.h"
#include "core/definition/CoreTypes.h"
#include "math/simd/SimdVectors.h"

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
		Ifrit::Math::SIMD::vfloat3 vz, vw;
		Ifrit::Math::SIMD::vfloat3 bx, by;//b1, b2, b3;
		Ifrit::Math::SIMD::vfloat4 f1, f2, f3; //Interpolate Bases
		Ifrit::Math::SIMD::vfloat3 e1, e2, e3; //Edge Coefs
		int originalPrimitive;
	}; 

	struct PendingTriangleProposalCUDA {
		ifloat4 v1, v2, v3;
		ifloat3 b1, b2, b3;
	};
	struct TriangleZW {
		float z, w;
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

	struct TileBinProposalCUDA {
		ishort2 tile;
		ishort2 tileEnd;
		int primId;
	};

	struct TilePixelProposalCUDA {
		ishort2 px;
		int primId;
	};

	struct TilePixelBitProposalCUDA {
		int mask;
		int primId;
	};

	struct TilePixelProposalExperimentalCUDA {
		int mask;
		int primId;
		int subTileId;
	};

	struct PrimitiveEdgeCoefs {
		ifloat3 coef[3];
	};

	class TileRasterDeviceConstants {
	public:
		int vertexStride = 3;
		int vertexCount;
		int attributeCount;
		int varyingCount;
		int frameBufferWidth;
		int frameBufferHeight;
		bool counterClockwise = false;
		int totalIndexCount;
	};

#if IFRIT_USE_CUDA
	struct TileRasterClipVertexCUDA {
		float3 barycenter;
		float4 pos;
	};
#endif

	struct AlphaBlendingCoefs {
		ifloat4 s, d;
	};
}