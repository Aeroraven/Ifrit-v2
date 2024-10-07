#pragma once

#include "core/definition/CoreExports.h"
#include "engine/tileraster/TileRasterRenderer.h"

namespace Ifrit::Engine::TileRaster {
	

	struct AssembledTriangleRef {
		int sourcePrimitive;
		int vertexReferences[3];
	};

	constexpr auto tagbufferSizeX = TileRasterContext::tileWidth;
	struct TagBufferContext {
		Ifrit::Math::SIMD::vfloat3 tagBufferBary[tagbufferSizeX * tagbufferSizeX];
		Ifrit::Math::SIMD::vfloat3 atpBx[tagbufferSizeX * tagbufferSizeX];
		Ifrit::Math::SIMD::vfloat3 atpBy[tagbufferSizeX * tagbufferSizeX];
		Ifrit::Math::SIMD::vfloat4 atpF1[tagbufferSizeX * tagbufferSizeX];
		Ifrit::Math::SIMD::vfloat4 atpF2[tagbufferSizeX * tagbufferSizeX];
		Ifrit::Math::SIMD::vfloat4 atpF3[tagbufferSizeX * tagbufferSizeX];
		int valid[tagbufferSizeX * tagbufferSizeX];
	};

	struct PixelShadingFuncArgs {
		ImageF32* depthAttachmentPtr;
		int varyingCounts;
		ImageF32* colorAttachment0;
		const int* indexBufferPtr;
		TagBufferContext* tagBuffer;
		bool forcedInQuads;
	};

	class TileRasterWorker {
	protected:
		std::atomic<TileRasterStage> status;
		std::atomic<bool> activated;
	private:
		uint32_t workerId;
		TileRasterRenderer* rendererReference; 
		std::unique_ptr<std::thread> execWorker;
		std::shared_ptr<TileRasterContext> context;
		
		std::vector<ifloat4> perVertexVaryings;
		std::vector<Ifrit::Math::SIMD::vfloat4> interpolatedVaryings;
		std::vector<const void*> interpolatedVaryingsAddr;
		std::vector<const void*> perVertexVaryingsAddr;

		std::vector<ifloat4> colorOutput = std::vector<ifloat4>(1);
		std::vector<AssembledTriangleProposal> generatedTriangle;

		// Hold Cache
		float depthCache[TileRasterContext::tileWidth * TileRasterContext::tileWidth];
		std::vector<TileBinProposal> coverQueueLocal;

		//Debug
		int totalDraws = 0;
		int reqDraws = 0;
		const float EPS = 1e-8;
		const float EPS2 = 1e-8;

		bool vert = false;
		bool geom = false;
		bool rast = false;
		bool frag = false;

	public:
		TileRasterWorker(uint32_t workerId, std::shared_ptr<TileRasterRenderer> renderer, std::shared_ptr<TileRasterContext> context);
		
	protected:
		friend class TileRasterRenderer;
		void run() IFRIT_AP_NOTHROW;
		void drawCall(bool withClear) IFRIT_AP_NOTHROW;
		void drawCallWithClear() IFRIT_AP_NOTHROW;
		void release();

		bool triangleFrustumClip(Ifrit::Math::SIMD::vfloat4 v1, Ifrit::Math::SIMD::vfloat4 v2, Ifrit::Math::SIMD::vfloat4 v3, Ifrit::Math::SIMD::vfloat4& bbox) IFRIT_AP_NOTHROW;
		uint32_t triangleHomogeneousClip(const int primitiveId, Ifrit::Math::SIMD::vfloat4 v1, Ifrit::Math::SIMD::vfloat4 v2, Ifrit::Math::SIMD::vfloat4 v3) IFRIT_AP_NOTHROW;
		bool triangleCulling(Ifrit::Math::SIMD::vfloat4 v1, Ifrit::Math::SIMD::vfloat4 v2, Ifrit::Math::SIMD::vfloat4 v3) IFRIT_AP_NOTHROW;
		void executeBinner(const int primitiveId, const AssembledTriangleProposalRasterStage& atp, Ifrit::Math::SIMD::vfloat4 bbox) IFRIT_AP_NOTHROW;

		void vertexProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW;
		void geometryProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW;
		void tiledProcessing(TileRasterRenderer* renderer, bool clearDepth) IFRIT_AP_NOTHROW;

		void rasterizationSingleTile(TileRasterRenderer* renderer, int tileId) IFRIT_AP_NOTHROW;
		void sortOrderProcessingSingleTile(TileRasterRenderer* renderer, int tileId) IFRIT_AP_NOTHROW;
		void fragmentProcessingSingleTile(TileRasterRenderer* renderer,bool clearedDepth, int tileId) IFRIT_AP_NOTHROW;

		void threadStart();

		template<bool tpAlphaBlendEnable,IfritCompareOp tpDepthFunc, bool tpOnlyTaggingPass>
		void pixelShading(const AssembledTriangleProposalShadeStage& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

		template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc, bool tpOnlyTaggingPass>
		void pixelShadingSIMD128(const AssembledTriangleProposalShadeStage& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;
		
		template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc, bool tpOnlyTaggingPass>
		void pixelShadingSIMD256(const AssembledTriangleProposalShadeStage& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

		void pixelShadingFromTagBuffer(const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

		inline int getTileID(int x, int y) IFRIT_AP_NOTHROW {
			return y * context->numTilesX + x;
		
		}
		
	};
}