#pragma once

#include "core/definition/CoreExports.h"
#include "engine/tileraster/TileRasterRenderer.h"

namespace Ifrit::Engine::TileRaster {
	struct VaryingPlaceholder {
		char p[64];
	};
	class TileRasterWorker {
	public:
		std::atomic<TileRasterStage> status;
		std::atomic<bool> activated;
	private:
		std::unique_ptr<std::thread> execWorker;
		uint32_t workerId;
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

		std::vector<float4> colorOutput = std::vector<float4>(1);

		const float EPS = 1e-7;
		const float EPS2 = 1e-7;

	public:
		TileRasterWorker(uint32_t workerId, std::shared_ptr<TileRasterRenderer> renderer, std::shared_ptr<TileRasterContext> context);
		void run();

		bool triangleFrustumClip(float4 v1, float4 v2, float4 v3, rect2Df& bbox);
		bool triangleCulling(float4 v1, float4 v2, float4 v3);
		void executeBinner(const int primitiveId, float4 v1, float4 v2, float4 v3, rect2Df bbox);

		void vertexProcessing();
		void geometryProcessing();
		void rasterization();
		void fragmentProcessing();

		void threadStart();
		void pixelShading(const int primitiveId, const int dx, const int dy);

		void interpolateVaryings(int id, const int indices[3], const float barycentric[3], VaryingStore& dest);
		void getVertexAttributes(const int id, std::vector<const void*>& out);
		void getVaryingsAddr(const int id,std::vector<VaryingStore*>& out);

		void pixelShadingSIMD128(const int primitiveId, const int dx, const int dy);
		void pixelShadingSIMD256(const int primitiveId, const int dx, const int dy);

		inline float edgeFunction(float4 a, float4 b, float4 c) {
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
		inline int getTileID(int x, int y) {
			return y * context->tileBlocksX + x;
		}
		inline void getAcceptRejectCoords(float3 edgeCoefs[3], int chosenCoordTR[3], int chosenCoordTA[3]) {
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