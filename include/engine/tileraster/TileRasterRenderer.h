#pragma once
#include "core/definition/CoreExports.h"
#include "core/data/Image.h"
#include "engine/base/FrameBuffer.h"
#include "engine/base/Renderer.h"
#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShader.h"
#include "engine/base/VertexShaderResult.h"

namespace Ifrit::Engine::TileRaster {
	using namespace Ifrit::Engine;
	class TileRasterRenderer : public Renderer {
	private:
		struct TileBinProposal {
			int primitiveId;
			rect2Df bbox;
			bool allAccept;
		};
	private:
		FrameBuffer* frameBuffer;
		const VertexBuffer* vertexBuffer;
		const std::vector<int>* indexBuffer;
		VertexShader* vertexShader;
		VertexShaderResult* vertexShaderResult;
		std::vector<std::vector<TileBinProposal>> rasterizerQueue;

		int numThreads = 8;
		int vertexStride = 3;
		int tileBlocksX = 16;

		

	public:

		void bindFrameBuffer(FrameBuffer& frameBuffer);
		void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		void bindIndexBuffer(const std::vector<int>& indexBuffer);
		void bindVertexShader(VertexShader& vertexShader);

		void render();
		bool triangleFrustumClip(float4 v1, float4 v2, float4 v3, rect2Df& bbox);
		bool triangleCulling(float4 v1, float4 v2, float4 v3);
		void executeBinner(const int threadId, const int primitiveId, float4 v1, float4 v2, float4 v3, rect2Df bbox);
	};
}