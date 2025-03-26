
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
#include "ifrit/softgraphics/core/data/Image.h"
#include "ifrit/softgraphics/engine/base/FrameBuffer.h"
#include "ifrit/softgraphics/engine/base/Renderer.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/base/VaryingDescriptor.h"
#include "ifrit/softgraphics/engine/base/VertexBuffer.h"
#include "ifrit/softgraphics/engine/base/VertexShaderResult.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterCommon.h"

namespace Ifrit::Graphics::SoftGraphics::TileRaster
{
	struct TileRasterContextRasterQueueProposal
	{
		u32 workerId;
		int primId;
	};
	struct AssembledTriangleProposalShadeStage
	{
		Ifrit::Math::SIMD::SVector4f f1, f2, f3; // Interpolate Bases
		Ifrit::Math::SIMD::SVector3f bx, by;
		int							 originalPrimitive;
	};
	struct AssembledTriangleProposalRasterStage
	{
		Ifrit::Math::SIMD::SVector3f e1, e2, e3; // Edge Coefs
	};

	class TileRasterContext
	{
	public:
		// Config
		IF_CONSTEXPR static int numThreads = 16;
		IF_CONSTEXPR static int vertexStride = 3;
		IF_CONSTEXPR static int tileWidth = 16;
		IF_CONSTEXPR static int subtileBlockWidth = 4;
		IF_CONSTEXPR static int numSubtilesPerTileX = tileWidth / subtileBlockWidth;
		IF_CONSTEXPR static int vsChunkSize = 48;
		IF_CONSTEXPR static int gsChunkSize = 128;

		IF_CONSTEXPR static int workerReprBits = 8;

		// Non-owning Bindings
		FrameBuffer*			frameBuffer;
		const VertexBuffer*		vertexBuffer;
		const int*				indexBuffer;
		int						indexBufferSize;
		VertexShader*			vertexShader;
		VaryingDescriptor*		varyingDescriptor;
		FragmentShader*			fragmentShader;
		std::unordered_map<std::pair<int, int>, const void*, Ifrit::Graphics::SoftGraphics::Core::Utility::PairHash>
														  uniformMapping;

		// Cached attributes
		int												  frameWidth;
		int												  frameHeight;
		float											  invFrameWidth;
		float											  invFrameHeight;

		// Owning Bindings
		std::unique_ptr<VaryingDescriptor>				  owningVaryingDesc;

		// Thread-safe Calls
		VertexShader*									  threadSafeVS[TileRasterContext::numThreads + 1];
		FragmentShader*									  threadSafeFS[TileRasterContext::numThreads + 1];

		std::unique_ptr<VertexShader>					  threadSafeVSOwningSection[TileRasterContext::numThreads + 1];
		std::unique_ptr<FragmentShader>					  threadSafeFSOwningSection[TileRasterContext::numThreads + 1];

		// Resources
		std::unique_ptr<VertexShaderResult>				  vertexShaderResult;
		std::vector<std::vector<int>>					  rasterizerQueue[TileRasterContext::numThreads + 1];
		std::vector<std::vector<int>>					  coverQueue[TileRasterContext::numThreads + 1];

		// Sorted List
		std::vector<std::vector<TileBinProposal>>		  sortedCoverQueue;
		int												  numTilesX = 1;
		int												  numTilesY = 1;

		TileRasterFrontFace								  frontface = TileRasterFrontFace::CLOCKWISE;
		IfritColorAttachmentBlendState					  blendState;
		AlphaBlendingCoefs								  blendColorCoefs;
		AlphaBlendingCoefs								  blendAlphaCoefs;
		IfritCompareOp									  depthFunc = IF_COMPARE_OP_LESS;
		IfritCompareOp									  depthFuncSaved = IF_COMPARE_OP_LESS;

		// Options
		bool											  optForceDeterministic = true;
		bool											  optDepthTestEnableII = true;

		// Geometry
		std::vector<AssembledTriangleProposalRasterStage> assembledTrianglesRaster[TileRasterContext::numThreads + 1];
		std::vector<AssembledTriangleProposalShadeStage>  assembledTrianglesShade[TileRasterContext::numThreads + 1];
	};
} // namespace Ifrit::Graphics::SoftGraphics::TileRaster