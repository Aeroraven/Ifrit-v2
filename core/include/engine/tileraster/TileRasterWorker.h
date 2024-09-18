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
		int valid[tagbufferSizeX * tagbufferSizeX];
	};

	struct PixelShadingFuncArgs {
		ImageF32* depthAttachmentPtr;
		int varyingCounts;
		ImageF32* colorAttachment0;
		const int* indexBufferPtr;
		TagBufferContext* tagBuffer;
	};

	class TileRasterWorker {
	protected:
		std::atomic<TileRasterStage> status;
		std::atomic<bool> activated;
	private:
		uint32_t workerId;
		//Hold refernce to the parent. weak_ptr is time-consuming. This section conveys no ownership semantic
		//worker object have the same lifetime with their parent
		TileRasterRenderer* rendererReference; 
		std::unique_ptr<std::thread> execWorker;
		std::shared_ptr<TileRasterContext> context;
		

		std::vector<ifloat4> perVertexVaryings;
		std::vector<Ifrit::Math::SIMD::vfloat4> interpolatedVaryings;
		std::vector<const void*> interpolatedVaryingsAddr;
		std::vector<const void*> perVertexVaryingsAddr;

		std::vector<ifloat4> colorOutput = std::vector<ifloat4>(1);
		std::vector<AssembledTriangleProposal> generatedTriangle;

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
		void executeBinner(const int primitiveId, const AssembledTriangleProposal& atp, Ifrit::Math::SIMD::vfloat4 bbox) IFRIT_AP_NOTHROW;

		void vertexProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW;
		void geometryProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW;
		void rasterization(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW;
		void sortOrderProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW;
		void fragmentProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW;

		void threadStart();

		void getVertexAttributes(const int id, std::vector<const void*>& out) IFRIT_AP_NOTHROW ;
		void getVaryingsAddr(const int id,std::vector<Ifrit::Math::SIMD::vfloat4*>& out)IFRIT_AP_NOTHROW ;

		template<bool tpAlphaBlendEnable,IfritCompareOp tpDepthFunc, bool tpOnlyTaggingPass>
		void pixelShading(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

		template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc, bool tpOnlyTaggingPass>
		void pixelShadingSIMD128(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;
		
		template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc, bool tpOnlyTaggingPass>
		void pixelShadingSIMD256(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

		void pixelShadingFromTagBuffer(const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

		inline int getTileID(int x, int y) IFRIT_AP_NOTHROW {
			return y * context->numTilesX + x;
		
		}
		
	};
}