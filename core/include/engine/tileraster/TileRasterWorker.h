#pragma once

#include "core/definition/CoreExports.h"
#include "engine/tileraster/TileRasterRenderer.h"

namespace Ifrit::Engine::TileRaster {
	

	struct AssembledTriangleRef {
		int sourcePrimitive;
		int vertexReferences[3];
	};

	struct PixelShadingFuncArgs {
		ImageF32* depthAttachmentPtr;
		int varyingCounts;
		ImageF32* colorAttachment0;
		const int* indexBufferPtr;
	};

	class TileRasterWorker {
	protected:
		std::atomic<TileRasterStage> status;
		std::atomic<bool> activated;
	private:
		uint32_t workerId;
		std::unique_ptr<std::thread> execWorker;
		std::shared_ptr<TileRasterRenderer> renderer;
		std::shared_ptr<TileRasterContext> context;
		bool vert= false;
		bool geom = false;
		bool rast = false;
		bool frag = false;

		std::vector<VaryingStore> perVertexVaryings;
		std::vector<VaryingStore> interpolatedVaryings;
		std::vector<const void*> interpolatedVaryingsAddr;
		std::vector<const void*> perVertexVaryingsAddr;

		std::vector<ifloat4> colorOutput = std::vector<ifloat4>(1);
		std::vector<AssembledTriangleProposal> generatedTriangle;

		const float EPS = 1e-8;
		const float EPS2 = 1e-8;

	public:
		TileRasterWorker(uint32_t workerId, std::shared_ptr<TileRasterRenderer> renderer, std::shared_ptr<TileRasterContext> context);

	protected:
		friend class TileRasterRenderer;
		void run() IFRIT_AP_NOTHROW;

		bool triangleFrustumClip(ifloat4 v1, ifloat4 v2, ifloat4 v3, irect2Df& bbox) IFRIT_AP_NOTHROW;
		uint32_t triangleHomogeneousClip(const int primitiveId, ifloat4 v1, ifloat4 v2, ifloat4 v3) IFRIT_AP_NOTHROW;
		bool triangleCulling(ifloat4 v1, ifloat4 v2, ifloat4 v3) IFRIT_AP_NOTHROW;
		void executeBinner(const int primitiveId, const AssembledTriangleProposal& atp, irect2Df bbox) IFRIT_AP_NOTHROW;

		void vertexProcessing() IFRIT_AP_NOTHROW;
		void geometryProcessing() IFRIT_AP_NOTHROW;
		void rasterization() IFRIT_AP_NOTHROW;
		void sortOrderProcessing() IFRIT_AP_NOTHROW;
		void fragmentProcessing() IFRIT_AP_NOTHROW;

		void threadStart();

		void interpolateVaryings(int id, const int indices[3], const float barycentric[3], VaryingStore& dest) IFRIT_AP_NOTHROW;
		void getVertexAttributes(const int id, std::vector<const void*>& out) IFRIT_AP_NOTHROW ;
		void getVaryingsAddr(const int id,std::vector<VaryingStore*>& out)IFRIT_AP_NOTHROW ;

		template<bool tpAlphaBlendEnable,IfritCompareOp tpDepthFunc>
		void pixelShading(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

		template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc>
		void pixelShadingSIMD128(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;
		
		template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc>
		void pixelShadingSIMD256(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

		inline float edgeFunction(ifloat4 a, ifloat4 b, ifloat4 c) {
			return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
		}

#ifdef IFRIT_USE_SIMD_128
		inline __m128 edgeFunctionSIMD128(__m128& aX, __m128& aY, __m128& bX, __m128& bY, __m128& cX, __m128& cY) {
			return _mm_sub_ps(_mm_mul_ps(_mm_sub_ps(cX, aX), _mm_sub_ps(bY, aY)), _mm_mul_ps(_mm_sub_ps(cY, aY), _mm_sub_ps(bX, aX)));
		}
#endif
#ifdef IFRIT_USE_SIMD_256
		inline __m256 edgeFunctionSIMD256(__m256& aX, __m256& aY, __m256& bX, __m256& bY, __m256& cX, __m256& cY) {
			return _mm256_sub_ps(_mm256_mul_ps(_mm256_sub_ps(cX, aX), _mm256_sub_ps(bY, aY)), _mm256_mul_ps(_mm256_sub_ps(cY, aY), _mm256_sub_ps(bX, aX)));
		}
#endif
		inline int getTileID(int x, int y) IFRIT_AP_NOTHROW {
			return y * context->numTilesX + x;
		
		}
		inline void getAcceptRejectCoords(ifloat3 edgeCoefs[3], int chosenCoordTR[3], int chosenCoordTA[3])IFRIT_AP_NOTHROW {
			constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;
			for (int i = 0; i < 3; i++) {
				bool normalRight = edgeCoefs[i].x < 0;
				bool normalDown = edgeCoefs[i].y < 0;
				if (normalRight) {
					if (normalDown) {
						chosenCoordTR[i] = VRB;
						chosenCoordTA[i] = VLT;
					}
					else {
						chosenCoordTR[i] = VRT;
						chosenCoordTA[i] = VLB;
					}
				}
				else {
					if (normalDown) {
						chosenCoordTR[i] = VLB;
						chosenCoordTA[i] = VRT;
					}
					else {
						chosenCoordTR[i] = VLT;
						chosenCoordTA[i] = VRB;
					}
				}
			}
		}
	};
}