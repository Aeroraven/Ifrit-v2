#pragma once

#include "core/data/Image.h"
#include "engine/base/FrameBuffer.h"
#include "engine/base/Renderer.h"
#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShader.h"
#include "engine/base/FragmentShader.h"
#include "engine/base/VertexShaderResult.h"

namespace Ifrit::Engine::TileRaster {
	enum class TileRasterLevel {
		TILE,
		BLOCK,
		PIXEL
	};

	struct TileBinProposal {
		int primitiveId;
		rect2Df bbox;
		int2 tile;
		bool allAccept;
		TileRasterLevel level;
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
		int numThreads = 6;
		int vertexStride = 3;
		int tileBlocksX = 32;
		int subtileBlocksX = 8;

	};
}