
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
#include "ifrit/softgraphics/core/definition/CoreTypes.h"

namespace Ifrit::Graphics::SoftGraphics::TileRaster
{
	enum class TileRasterHomogeneousClipping
	{
		DISABLE,
		SIMPLE_DISCARD,
		HOMOGENEOUS_CLIPPING
	};

	enum class TileRasterFrontFace
	{
		CLOCKWISE,
		COUNTER_CLOCKWISE
	};

	enum class TileRasterLevel
	{
		TILE,
		BLOCK,
		PIXEL,
		PIXEL_PACK2X2, // SIMD128
		PIXEL_PACK4X2, // SIMD256
		PIXEL_PACK4X4, // SIMD512
	};

	struct AssembledTriangleProposal
	{
		Ifrit::Math::SIMD::SVector3f vz, vw;
		Ifrit::Math::SIMD::SVector3f bx, by;	 // b1, b2, b3;
		Ifrit::Math::SIMD::SVector4f f1, f2, f3; // Interpolate Bases
		Ifrit::Math::SIMD::SVector3f e1, e2, e3; // Edge Coefs
		int							 originalPrimitive;
	};

	struct PendingTriangleProposalCUDA
	{
		Vector4f v1, v2, v3;
		Vector3f b1, b2, b3;
	};
	struct TriangleZW
	{
		float z, w;
	};

	struct AssembledTriangleProposalReference
	{
		u32 workerId;
		int primId;
	};

	struct TileBinProposal
	{
		u64				tile;
		TileRasterLevel level;
		int				clipTrianglePacked;
	};

	struct TileBinProposalCUDA
	{
		Vector2s tile;
		Vector2s tileEnd;
		int		 primId;
	};

	struct TilePixelProposalCUDA
	{
		Vector2s px;
		int		 primId;
	};

	struct TilePixelBitProposalCUDA
	{
		int mask;
		int primId;
	};

	struct TilePixelProposalExperimentalCUDA
	{
		int mask;
		int primId;
		int subTileId;
	};

	struct PrimitiveEdgeCoefs
	{
		Vector3f coef[3];
	};

	class TileRasterDeviceConstants
	{
	public:
		int	 vertexStride = 3;
		int	 vertexCount;
		int	 attributeCount;
		int	 varyingCount;
		int	 frameBufferWidth;
		int	 frameBufferHeight;
		bool counterClockwise = false;
		int	 totalIndexCount;
	};

#if IFRIT_USE_CUDA
	struct TileRasterClipVertexCUDA
	{
		float3 barycenter;
		float4 pos;
	};
#endif

	struct AlphaBlendingCoefs
	{
		Vector4f s, d;
	};
} // namespace Ifrit::Graphics::SoftGraphics::TileRaster