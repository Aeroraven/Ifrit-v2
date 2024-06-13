#include "engine/tilerastercuda/TileRasterInvocationCuda.cuh"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterConstantsCuda.h"
#include <cuda_profiler_api.h>

namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	IFRIT_DEVICE_CONST static int csFrameWidth = 0;
	IFRIT_DEVICE_CONST static int csFrameHeight = 0;
	IFRIT_DEVICE_CONST static float csFrameWidthInv = 0;
	IFRIT_DEVICE_CONST static float csFrameHeightInv = 0;
	IFRIT_DEVICE_CONST static bool csCounterClosewiseCull = false;
	IFRIT_DEVICE_CONST static int csVertexOffsets[CU_MAX_ATTRIBUTES];
	IFRIT_DEVICE_CONST static int csTotalVertexOffsets = 0;
	IFRIT_DEVICE_CONST static int csVaryingCounts = 0;
	IFRIT_DEVICE_CONST static int csAttributeCounts = 0;
	IFRIT_DEVICE_CONST static int csVertexCount = 0;
	IFRIT_DEVICE_CONST static int csTotalIndices = 0;


	static int hsFrameWidth = 0;
	static int hsFrameHeight = 0;
	static bool hsCounterClosewiseCull = false;
	static int hsVertexOffsets[CU_MAX_ATTRIBUTES];
	static int hsTotalVertexOffsets = 0;
	static int hsVaryingCounts = 0;
	static int hsAttributeCounts = 0;
	static int hsVertexCounts = 0;
	static int hsTotalIndices = 0;

	template<class T>
	class LocalDynamicVector {
	public:
		T* data[CU_VECTOR_HIERARCHY_LEVEL];
		int size;
		int lock;
	public:
		IFRIT_DEVICE void initialize() {
			int baseSize = CU_VECTOR_BASE_LENGTH;
			int capacity = (1 << baseSize);
			size = 0;
			data[0] = (T*)malloc(sizeof(T) * capacity);
			data[1] = (T*)malloc(sizeof(T) * capacity * 2);
			lock = 0;
			if(data[0] == nullptr || data[1] == nullptr) {
				printf("ERROR: MALLOC FAILED INIT\n");
				asm("trap;");
			}
			for (int i = 2; i < CU_VECTOR_HIERARCHY_LEVEL; i++) {
				data[i] = nullptr;
			}
		}
		
		IFRIT_DEVICE void push_back(const T& x) {
			auto idx = atomicAdd(&size, 1);
			int putBucket = max(0, (31 - (CU_VECTOR_BASE_LENGTH - 1) - __clz(idx)));
			if (putBucket) {
				int pralloc = putBucket;
				if (data[pralloc] == nullptr) {
					while (atomicCAS(&lock, 0, 1) != 0) {}
					if (data[pralloc] == nullptr) {
						data[pralloc] = (T*)malloc(sizeof(T) * (1 << ((CU_VECTOR_BASE_LENGTH - 1) + pralloc)));
						if (data[pralloc] == nullptr) {
							printf("ERROR: MALLOC FAILED\n");
							asm("trap;");
						}
					}
					atomicExch(&lock, 0);
				}
			}
			int prevLevel = putBucket ? (1 << (CU_VECTOR_BASE_LENGTH-1+putBucket)) : 0;
			data[putBucket][idx - prevLevel] = x;
		}
		IFRIT_DEVICE int reserve_back(int v) {
			auto idx = atomicAdd(&size, v);
			auto idxs = idx;
			idx += v - 1;
			int putBucket = max(0, (31 - (CU_VECTOR_BASE_LENGTH - 1) - __clz(idx)));
			if (putBucket) {
				int pralloc = putBucket;
				if (data[pralloc] == nullptr) {
					while (atomicCAS(&lock, 0, 1) != 0) {}
					if (data[pralloc] == nullptr) {
						data[pralloc] = (T*)malloc(sizeof(T) * (1 << ((CU_VECTOR_BASE_LENGTH - 1) + pralloc)));
						if (data[pralloc] == nullptr) {
							printf("ERROR: MALLOC FAILED\n");
							asm("trap;");
						}
					}
					atomicExch(&lock, 0);
				}
			}
			return idxs;
		}

		IFRIT_DEVICE void write(int v, const T& x) {
			int putBucket = max(0, (31 - __clz(v)) - (CU_VECTOR_BASE_LENGTH - 1));
			int prevLevel = putBucket ? (1 << (CU_VECTOR_BASE_LENGTH - 1 + putBucket)) : 0;
			data[putBucket][v - prevLevel] = x;
		}

		IFRIT_DEVICE T at(int idx) {
			int putBucket = max(0, (31 - __clz(idx)) - (CU_VECTOR_BASE_LENGTH-1));
			int prevLevel = putBucket ? (1 << (CU_VECTOR_BASE_LENGTH - 1 + putBucket)) : 0;
			return data[putBucket][idx - prevLevel];
		}
		IFRIT_DEVICE int getSize() {
			return size;
		}
		

		IFRIT_DEVICE int clear() {
			size = 0;
		}
		IFRIT_DEVICE void setSize(int newSize) {
			size = newSize;
		}
	};
	
	IFRIT_DEVICE static LocalDynamicVector<uint32_t> dRasterQueueM2[CU_BIN_SIZE * CU_BIN_SIZE];
	IFRIT_DEVICE static LocalDynamicVector<uint32_t> dCoverQueueFullM2[CU_BIN_SIZE * CU_BIN_SIZE];
	IFRIT_DEVICE static LocalDynamicVector<uint32_t> dCoverQueueSuperTileFullM2[CU_LARGE_BIN_SIZE * CU_LARGE_BIN_SIZE];
	IFRIT_DEVICE static LocalDynamicVector<TilePixelBitProposalCUDA> dCoverQueuePixelM2[CU_TILE_SIZE * CU_TILE_SIZE][CU_MAX_SUBTILES_PER_TILE];

	IFRIT_DEVICE static uint32_t dRasterQueueWorklistPrim[CU_BIN_SIZE * 2 * CU_SINGLE_TIME_TRIANGLE];
	IFRIT_DEVICE static uint32_t dRasterQueueWorklistTile[CU_BIN_SIZE * 2 * CU_SINGLE_TIME_TRIANGLE];
	IFRIT_DEVICE static uint32_t dRasterQueueWorklistCounter;

	IFRIT_DEVICE static irect2Df dBoundingBoxM2[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static AssembledTriangleProposalCUDA dAssembledTriangleM2[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static uint32_t dAssembledTriangleCounterM2;
	IFRIT_DEVICE static float dHierarchicalDepthTile[CU_TILE_SIZE * CU_TILE_SIZE];

	// Profiler: Overdraw
	IFRIT_DEVICE static int dOverDrawCounter;
	IFRIT_DEVICE static int dOverZTestCounter;
	IFRIT_DEVICE static int dIrrelevantPixel;
	IFRIT_DEVICE static int dTotalBinnedPixel;

	// Profiler: Second Binner Utilizations
	IFRIT_DEVICE static int dSecondBinnerTotalReqs;
	IFRIT_DEVICE static int dSecondBinnerActiveReqs;
	IFRIT_DEVICE static int dSecondBinnerEmptyTiles;

	IFRIT_DEVICE float devEdgeFunction(ifloat4 a, ifloat4 b, ifloat4 c) {
		return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
	}
	IFRIT_DEVICE bool devTriangleCull(ifloat4 v1, ifloat4 v2, ifloat4 v3) {
		float a1 = 1.0f;
		float a2 = 1.0f;
		float a3 = 1.0f;
		float d1 = (v1.x * v2.y) * a1 * a2;
		float d2 = (v2.x * v3.y) * a2 * a3;
		float d3 = (v3.x * v1.y) * a3 * a1;
		float n1 = (v3.x * v2.y) * a3 * a2;
		float n2 = (v1.x * v3.y) * a1 * a3;
		float n3 = (v2.x * v1.y) * a2 * a1;
		float d = d1 + d2 + d3 - n1 - n2 - n3;
		return d > 0.0f;
	}

	IFRIT_DEVICE bool devViewSpaceClip(ifloat4 v1, ifloat4 v2, ifloat4 v3) {
		auto maxX = max(v1.x, max(v2.x, v3.x));
		auto maxY = max(v1.y, max(v2.y, v3.y));
		auto minX = min(v1.x, min(v2.x, v3.x));
		auto minY = min(v1.y, min(v2.y, v3.y));
		auto isIllegal = maxX < 0.0f || maxY < 0.0f || minX > 1.0f || minY > 1.0f;
		return isIllegal;
	}

	IFRIT_DEVICE void devGetAcceptRejectCoords(ifloat3 edgeCoefs[3], int chosenCoordTR[3], int chosenCoordTA[3]) {
		constexpr const int VLB = 0, VLT = 1, VRT = 3, VRB = 2;
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

	IFRIT_DEVICE bool devGetBBox(const ifloat4& v1, const ifloat4& v2, const ifloat4& v3, irect2Df& bbox) {
		bool inside = true;
		float minx = min(v1.x, min(v2.x, v3.x));
		float miny = min(v1.y, min(v2.y, v3.y));
		float maxx = max(v1.x, max(v2.x, v3.x));
		float maxy = max(v1.y, max(v2.y, v3.y));
		float maxz = max(v1.z, max(v2.z, v3.z));
		float minz = min(v1.z, min(v2.z, v3.z));
		bbox.x = minx;
		bbox.y = miny;
		bbox.w = maxx;
		bbox.h = maxy;
		return true;
	}

	IFRIT_DEVICE void devExecuteBinnerLargeTile(int primitiveId, const AssembledTriangleProposalCUDA& atp, irect2Df bbox) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;
		float minx = bbox.x;
		float miny = bbox.y;
		float maxx = bbox.w;
		float maxy = bbox.h;

		int tileLargeMinx = max(0, (int)(minx * CU_LARGE_BIN_SIZE));
		int tileLargeMiny = max(0, (int)(miny * CU_LARGE_BIN_SIZE));
		int tileLargeMaxx = min(CU_LARGE_BIN_SIZE - 1, (int)(maxx * CU_LARGE_BIN_SIZE));
		int tileLargeMaxy = min(CU_LARGE_BIN_SIZE - 1, (int)(maxy * CU_LARGE_BIN_SIZE));

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = atp.e1;
		edgeCoefs[1] = atp.e2;
		edgeCoefs[2] = atp.e3;

		ifloat2 tileCoordsT[4];
		ifloat2 tileCoordsS[4];

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		auto frameBufferWidth = csFrameWidth;
		auto frameBufferHeight = csFrameHeight;
		devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);
		for (int y = tileLargeMiny+threadIdx.y; y <= tileLargeMaxy; y+= CU_FIRST_BINNER_STRIDE) {

			auto curTileLargeY = y * frameBufferHeight / CU_LARGE_BIN_SIZE;
			auto curTileLargeY2 = (y + 1) * frameBufferHeight / CU_LARGE_BIN_SIZE;
			auto cty1 = 1.0f * curTileLargeY, cty2 = 1.0f * (curTileLargeY2 - 1);

			float criteriaTRLocalY[3];
			float criteriaTALocalY[3];

#define getY(w) (((w)&1)?cty1:cty2)
			for (int i = 0; i < 3; i++) {
				criteriaTRLocalY[i] = edgeCoefs[i].y * getY(chosenCoordTR[i]);
				criteriaTALocalY[i] = edgeCoefs[i].y * getY(chosenCoordTA[i]);
			}
#undef getY
			for (int x = tileLargeMinx + threadIdx.z; x <= tileLargeMaxx; x += CU_FIRST_BINNER_STRIDE) {

				auto curTileLargeX = x * frameBufferWidth / CU_LARGE_BIN_SIZE;
				auto curTileLargeX2 = (x + 1) * frameBufferWidth / CU_LARGE_BIN_SIZE;
				auto ctx1 = 1.0f * curTileLargeX;
				auto ctx2 = 1.0f * (curTileLargeX2 - 1);

				int criteriaTR = 0, criteriaTA = 0;

#define getX(w) (((w)>>1)?ctx2:ctx1)
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * getX(chosenCoordTR[i]) + criteriaTRLocalY[i];
					float criteriaTALocal = edgeCoefs[i].x * getX(chosenCoordTA[i]) + criteriaTALocalY[i];
					criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
					criteriaTA += criteriaTALocal < edgeCoefs[i].z;
				}
#undef getX

				if (criteriaTR != 3)
					continue;
				auto tileLargeId = y * CU_LARGE_BIN_SIZE + x;
				if (criteriaTA == 3) {
					dCoverQueueSuperTileFullM2[tileLargeId].push_back(primitiveId);
				}
				else {
					for (int dx = 0; dx < CU_BINS_PER_LARGE_BIN; dx++) {
						for (int dy = 0; dy < CU_BINS_PER_LARGE_BIN; dy++) {
							int ty = y * CU_BINS_PER_LARGE_BIN + dy;
							int tx = x * CU_BINS_PER_LARGE_BIN + dx;
							auto curTileY = ty * frameBufferHeight / CU_BIN_SIZE;
							auto curTileY2 = (ty + 1) * frameBufferHeight / CU_BIN_SIZE;
							auto cty1 = 1.0f * curTileY, cty2 = 1.0f * (curTileY2 - 1);
							auto curTileX = tx * frameBufferWidth / CU_BIN_SIZE;
							auto curTileX2 = (tx + 1) * frameBufferWidth / CU_BIN_SIZE;
							auto ctx1 = 1.0f * curTileX;
							auto ctx2 = 1.0f * (curTileX2 - 1);

							int criteriaTR = 0, criteriaTA = 0;
#define getX(w) (((w)>>1)?ctx2:ctx1)
#define getY(w) (((w)&1)?cty1:cty2)
							for (int i = 0; i < 3; i++) {
								float criteriaTRLocal = edgeCoefs[i].x * getX(chosenCoordTR[i]) + edgeCoefs[i].y * getY(chosenCoordTR[i]);
								float criteriaTALocal = edgeCoefs[i].x * getX(chosenCoordTA[i]) + edgeCoefs[i].y * getY(chosenCoordTA[i]);
								criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
								criteriaTA += criteriaTALocal < edgeCoefs[i].z;
							}
#undef getX
#undef getY
							if (criteriaTR != 3)
								continue;
							auto tileId = ty * CU_BIN_SIZE + tx;
							if (criteriaTA == 3) {
								dCoverQueueFullM2[tileId].push_back(primitiveId);
							}
							else {
								if constexpr (CU_OPT_WORKQUEUE_SECOND_BINNER) {
									auto plId = atomicAdd(&dRasterQueueWorklistCounter, 1);
									dRasterQueueWorklistTile[plId] = tileId;
									dRasterQueueWorklistPrim[plId] = primitiveId;
								}
								else {
									dRasterQueueM2[tileId].push_back(primitiveId);
								}
								
							}
								
						}
					}


				}
			}
		}
	}

	IFRIT_DEVICE void devExecuteBinner(int primitiveId, const AssembledTriangleProposalCUDA& atp,irect2Df bbox) {
		constexpr const int VLB = 0, VLT = 1, VRT = 3, VRB = 2;
		float minx = bbox.x;
		float miny = bbox.y;
		float maxx = bbox.w;
		float maxy = bbox.h;

		int tileMinx = max(0, (int)(minx * CU_BIN_SIZE));
		int tileMiny = max(0, (int)(miny * CU_BIN_SIZE));
		int tileMaxx = min(CU_BIN_SIZE - 1, (int)(maxx * CU_BIN_SIZE));
		int tileMaxy = min(CU_BIN_SIZE - 1, (int)(maxy * CU_BIN_SIZE));

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = atp.e1;
		edgeCoefs[1] = atp.e2;
		edgeCoefs[2] = atp.e3;

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		auto frameBufferWidth = csFrameWidth;
		auto frameBufferHeight = csFrameHeight;
		devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

		//printf("BBox: X=%d->%d, Y=%d->%d\n", tileMiny, tileMaxy, tileMinx, tileMaxx);
		for (int y = tileMiny+threadIdx.y; y <= tileMaxy; y += CU_FIRST_BINNER_STRIDE) {

			auto curTileY = y * frameBufferHeight / CU_BIN_SIZE;
			auto curTileY2 = (y + 1) * frameBufferHeight / CU_BIN_SIZE;
			auto cty1 = 1.0f * curTileY,cty2 = 1.0f * (curTileY2 - 1);

			float criteriaTRLocalY[3];
			float criteriaTALocalY[3];
#define getY(w) (((w)&1)?cty1:cty2)
			for(int i = 0; i < 3; i++) {
				criteriaTRLocalY[i] = edgeCoefs[i].y * getY(chosenCoordTR[i]);
				criteriaTALocalY[i] = edgeCoefs[i].y * getY(chosenCoordTA[i]);
			}
#undef getY
			for (int x = tileMinx+threadIdx.z; x <= tileMaxx; x += CU_FIRST_BINNER_STRIDE) {
				auto curTileX = x * frameBufferWidth / CU_BIN_SIZE;
				auto curTileX2 = (x + 1) * frameBufferWidth / CU_BIN_SIZE;
				auto ctx1 = 1.0f * curTileX;
				auto ctx2 = 1.0f * (curTileX2-1);
				int criteriaTR = 0,criteriaTA = 0;

#define getX(w) (((w)>>1)?ctx2:ctx1)
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * getX(chosenCoordTR[i]) + criteriaTRLocalY[i];
					float criteriaTALocal = edgeCoefs[i].x * getX(chosenCoordTA[i]) + criteriaTALocalY[i];
					criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
					criteriaTA += criteriaTALocal < edgeCoefs[i].z;
				}
#undef getX

				if (criteriaTR != 3) 
					continue;
				auto tileId = y * CU_BIN_SIZE + x;
				if (criteriaTA == 3) {
					dCoverQueueFullM2[tileId].push_back(primitiveId);
				}	
				else {
					if constexpr (CU_OPT_WORKQUEUE_SECOND_BINNER) {
						auto plId = atomicAdd(&dRasterQueueWorklistCounter, 1);
						dRasterQueueWorklistTile[plId] = tileId;
						dRasterQueueWorklistPrim[plId] = primitiveId;
					}
					else {
						dRasterQueueM2[tileId].push_back(primitiveId);
					}
					
				}
			}
		}
	}

	
	IFRIT_DEVICE inline void devInterpolateVaryings(
		int id,
		const VaryingStore* const* dVaryingBuffer,
		const int indices[3],
		const float barycentric[3],
		VaryingStore& dest
	) {
		const auto va = dVaryingBuffer[id];
		VaryingStore vd;
		vd.vf4 = { 0,0,0,0 };
		for (int j = 0; j < 3; j++) {
			auto vaf4 = va[indices[j]].vf4;
			vd.vf4.x += vaf4.x * barycentric[j];
			vd.vf4.y += vaf4.y * barycentric[j];
			vd.vf4.z += vaf4.z * barycentric[j];
			vd.vf4.w += vaf4.w * barycentric[j];
		}
		dest = vd;
	}

	IFRIT_DEVICE void devTilingRasterizationChildProcess(
		uint32_t tileIdX,
		uint32_t tileIdY,
		uint32_t primitiveSrcId,
		uint32_t subTileId,
		ifloat3 e1,
		ifloat3 e2,
		ifloat3 e3
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 3, VRB = 2;

		const auto tileId = tileIdY * CU_TILE_SIZE + tileIdX;
		const auto binId = tileIdY / CU_TILES_PER_BIN * CU_BIN_SIZE + tileIdX / CU_TILES_PER_BIN;

		const auto frameWidth = csFrameWidth;
		const auto frameHeight =csFrameHeight;

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = e1;
		edgeCoefs[1] = e2;
		edgeCoefs[2] = e3;
		
		auto curTileX = tileIdX * frameWidth / CU_TILE_SIZE;
		auto curTileY = tileIdY * frameHeight / CU_TILE_SIZE;
		auto curTileX2 = (tileIdX + 1) * frameWidth / CU_TILE_SIZE;
		auto curTileY2 = (tileIdY + 1) * frameHeight / CU_TILE_SIZE;
		auto curTileWid = curTileX2 - curTileX;
		auto curTileHei = curTileY2 - curTileY;


		// Decomp into Sub Blocks

		int numSubtilesX = curTileWid / CU_EXPERIMENTAL_SUBTILE_WIDTH + (curTileWid % CU_EXPERIMENTAL_SUBTILE_WIDTH != 0);
		int numSubtilesY = curTileHei / CU_EXPERIMENTAL_SUBTILE_WIDTH + (curTileHei % CU_EXPERIMENTAL_SUBTILE_WIDTH != 0);
		bool isFitX = (curTileWid % CU_EXPERIMENTAL_SUBTILE_WIDTH == 0);
		bool isFitY = (curTileHei % CU_EXPERIMENTAL_SUBTILE_WIDTH == 0);

		const auto numLim = numSubtilesX * numSubtilesY;
		
		const auto dq = dCoverQueuePixelM2[tileId];

		if(subTileId>=numLim) return;

		auto subTileIX = subTileId % numSubtilesX;
		auto subTileIY = subTileId / numSubtilesX;

		int criteriaTR = 0;
		int criteriaTA = 0;

		int subTilePixelX = curTileX + subTileIX * CU_EXPERIMENTAL_SUBTILE_WIDTH;
		int subTilePixelY = curTileY + subTileIY * CU_EXPERIMENTAL_SUBTILE_WIDTH;
		int subTilePixelX2 = curTileX + (subTileIX + 1) * CU_EXPERIMENTAL_SUBTILE_WIDTH;
		int subTilePixelY2 = curTileY + (subTileIY + 1) * CU_EXPERIMENTAL_SUBTILE_WIDTH;

		subTilePixelX2 = min(subTilePixelX2, curTileX2);
		subTilePixelY2 = min(subTilePixelY2, curTileY2);

		float subTileMinX = 1.0f * subTilePixelX;
		float subTileMinY = 1.0f * subTilePixelY;
		float subTileMaxX = 1.0f * (subTilePixelX2 - 1);
		float subTileMaxY = 1.0f * (subTilePixelY2 - 1);

		float ccTRX[3], ccTRY[3], ccTAX[3], ccTAY[3];

		#pragma unroll
		for (int i = 0; i < 3; i++) {
			bool normalRight = edgeCoefs[i].x < 0;
			bool normalDown = edgeCoefs[i].y < 0;
			if (normalRight) {
				if (normalDown) {
					ccTRX[i] = subTileMaxX; //RB
					ccTRY[i] = subTileMaxY; //RB
					ccTAX[i] = subTileMinX; //LT
					ccTAY[i] = subTileMinY; //LT
				}
				else {
					ccTRX[i] = subTileMaxX;
					ccTRY[i] = subTileMinY;
					ccTAX[i] = subTileMinX;
					ccTAY[i] = subTileMaxY;
				}
			}
			else {
				if (normalDown) {
					ccTRX[i] = subTileMinX;
					ccTRY[i] = subTileMaxY;
					ccTAX[i] = subTileMaxX;
					ccTAY[i] = subTileMinY;
				}
				else {
					ccTRX[i] = subTileMinX;
					ccTRY[i] = subTileMinY;
					ccTAX[i] = subTileMaxX;
					ccTAY[i] = subTileMaxY;
				}
			}
		}
		#pragma unroll
		for (int k = 0; k < 3; k++) {
			float criteriaTRLocal = edgeCoefs[k].x * ccTRX[k] + edgeCoefs[k].y * ccTRY[k];
			float criteriaTALocal = edgeCoefs[k].x * ccTAX[k] + edgeCoefs[k].y * ccTAY[k];
			criteriaTR += criteriaTRLocal < edgeCoefs[k].z;
			criteriaTA += criteriaTALocal < edgeCoefs[k].z;
		}

		if (criteriaTR != 3) {
			return;
		}
	
		int mask = 0;
		if (criteriaTA == 3) {
			mask |= ((1 << CU_EXPERIMENTAL_PIXELS_PER_SUBTILE) - 1);
		}
		else {
			//Into Pixel level
			int wid = subTilePixelX2 - subTilePixelX;
			float criteriaY[3];
			float criteriaX[3];

			const auto rsX1 = subTilePixelX * edgeCoefs[0].x;
			const auto rsX2 = subTilePixelX * edgeCoefs[1].x;
			const auto rsX3 = subTilePixelX * edgeCoefs[2].x;

			criteriaY[0] = subTilePixelY * edgeCoefs[0].y;
			criteriaY[1] = subTilePixelY * edgeCoefs[1].y;
			criteriaY[2] = subTilePixelY * edgeCoefs[2].y;
			criteriaX[0] = rsX1;
			criteriaX[1] = rsX2;
			criteriaX[2] = rsX3;

			#pragma unroll
			for (int i2 = 0; i2 < CU_EXPERIMENTAL_PIXELS_PER_SUBTILE; i2++) {
				auto dvX = i2 % CU_EXPERIMENTAL_SUBTILE_WIDTH;

				bool accept1 = (criteriaX[0] + criteriaY[0]) < edgeCoefs[0].z;
				bool accept2 = (criteriaX[1] + criteriaY[1]) < edgeCoefs[1].z;
				bool accept3 = (criteriaX[2] + criteriaY[2]) < edgeCoefs[2].z;

				int cond = (accept1 && accept2 && accept3 && dvX < wid);
				mask |= (cond << i2);

				if ((i2 + 1) % CU_EXPERIMENTAL_SUBTILE_WIDTH == 0) {
					criteriaY[0] += edgeCoefs[0].y;
					criteriaY[1] += edgeCoefs[1].y;
					criteriaY[2] += edgeCoefs[2].y;
					criteriaX[0] = rsX1;
					criteriaX[1] = rsX2;
					criteriaX[2] = rsX3;
				}
				else {
					criteriaX[0] += edgeCoefs[0].x;
					criteriaX[1] += edgeCoefs[1].x;
					criteriaX[2] += edgeCoefs[2].x;
				}
				
			}
		}
		TilePixelBitProposalCUDA nprop;
		nprop.mask = mask;
		nprop.primId = primitiveSrcId;
		dq[subTileId].push_back(nprop);
			
		
	}
 
	// Kernel Implementations

	IFRIT_KERNEL void vertexProcessingKernel(
		VertexShader* vertexShader,
		uint32_t vertexCount,
		char* dVertexBuffer,
		TypeDescriptorEnum* dVertexTypeDescriptor,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		ifloat4* dPosBuffer
	) {
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (globalInvoIdx >= vertexCount) return;
		const auto numAttrs = csAttributeCounts;
		const auto numVaryings = csVaryingCounts;

		const void* vertexInputPtrs[CU_MAX_ATTRIBUTES];
		VaryingStore* varyingOutputPtrs[CU_MAX_VARYINGS];
		
		for (int i = 0; i < numAttrs; i++) {
			vertexInputPtrs[i] = globalInvoIdx * csTotalVertexOffsets + dVertexBuffer + csVertexOffsets[i];
		}
		for (int i = 0; i < numVaryings; i++) {
			varyingOutputPtrs[i] = dVaryingBuffer[i] + globalInvoIdx;
		}
		auto s = *reinterpret_cast<const ifloat4*>(vertexInputPtrs[0]);
		vertexShader->execute(vertexInputPtrs, &dPosBuffer[globalInvoIdx], varyingOutputPtrs);
	}

	IFRIT_KERNEL void geometryProcessingKernel(
		ifloat4* IFRIT_RESTRICT_CUDA dPosBuffer,
		int* IFRIT_RESTRICT_CUDA dIndexBuffer,
		uint32_t startingIndexId,
		uint32_t indexCount
	) {
		
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if(globalInvoIdx >= indexCount / CU_TRIANGLE_STRIDE) return;

		const auto indexStart = globalInvoIdx * CU_TRIANGLE_STRIDE + startingIndexId;
		ifloat4 v1 = dPosBuffer[dIndexBuffer[indexStart]];
		ifloat4 v2 = dPosBuffer[dIndexBuffer[indexStart + 1]];
		ifloat4 v3 = dPosBuffer[dIndexBuffer[indexStart + 2]];
		
		if (csCounterClosewiseCull) {
			ifloat4 temp = v1;
			v1 = v3;
			v3 = temp;
		}
		const auto primId = globalInvoIdx + startingIndexId / CU_TRIANGLE_STRIDE;
		
		if (v1.w < 0 && v2.w < 0 && v3.w < 0) {
			return;
		}

		using Ifrit::Engine::Math::ShaderOps::CUDA::dot;
		using Ifrit::Engine::Math::ShaderOps::CUDA::sub;
		using Ifrit::Engine::Math::ShaderOps::CUDA::add;
		using Ifrit::Engine::Math::ShaderOps::CUDA::multiply;
		using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;

		constexpr uint32_t clipIts = (CU_OPT_HOMOGENEOUS_CLIPPING_NEG_W_ONLY) ? 1 : 7;
		constexpr int possibleTris = (CU_OPT_HOMOGENEOUS_CLIPPING_NEG_W_ONLY) ? 5 : 7;

		IFRIT_SHARED TileRasterClipVertexCUDA retd[2 * CU_GEOMETRY_PROCESSING_THREADS];
		IFRIT_SHARED int retdIndex[5 * CU_GEOMETRY_PROCESSING_THREADS];

		int retdTriCnt = 3;
		const auto retOffset = threadIdx.x * 2;
		const auto retIdxOffset = threadIdx.x * 5;
#define ret(fx,fy) retd[retdIndex[(fx)* possibleTris+(fy)+ retIdxOffset] + retOffset-3]
		
		uint32_t retCnt[2] = { 0,3 };

		int clipTimes = 0;
	
		ifloat4 outNormal = { 0,0,0,-1 };
		ifloat4 refPoint = { 0,0,0,1 };
		const auto cIdx = 0, cRIdx = 1;
		retCnt[cIdx] = 0;
		const auto psize = 3;

		TileRasterClipVertexCUDA pc;
		pc.barycenter = { 1,0,0 };
		pc.pos = v1;
		auto npc = dot(pc.pos, outNormal);

		ifloat4 vSrc[3] = { v1,v2,v3 };
		for (int j = 0; j < 3; j++) {
			auto pnPos = vSrc[(j + 1) % 3];
			auto npn = dot(pnPos, outNormal);
			
			ifloat3 pnBary = { 0,0,0 };
			if (j == 0) pnBary.y = 1;
			if (j == 1) pnBary.z = 1;
			if (j == 2) pnBary.x = 1;

			if constexpr (!CU_OPT_HOMOGENEOUS_DISCARD) {
				if (npc * npn < 0) {
					ifloat4 dir = sub(pnPos, pc.pos);
					float numo = pc.pos.w - pc.pos.x * refPoint.x - pc.pos.y * refPoint.y - pc.pos.z * refPoint.z;
					float deno = dir.x * refPoint.x + dir.y * refPoint.y + dir.z * refPoint.z - dir.w;
					float t = (-refPoint.w + numo) / deno;
					ifloat4 intersection = add(pc.pos, multiply(dir, t));
					ifloat3 barycenter = lerp(pc.barycenter, pnBary, t);

					TileRasterClipVertexCUDA newp;
					newp.barycenter = barycenter;
					newp.pos = intersection;
					retd[retdTriCnt + retOffset - 3] = newp;
					retdIndex[retCnt[cIdx] + retIdxOffset] = retdTriCnt;
					retCnt[cIdx]++;
					retdTriCnt++;
				}
			}
				
			if (npn < 0) {
				retdIndex[retCnt[cIdx] + retIdxOffset] = (j + 1) % 3;
				retCnt[cIdx]++;
			}
			npc = npn;
			pc.pos = pnPos;
			pc.barycenter = pnBary;
		}
		if (retCnt[cIdx] < 3) {
			return;
		}
		


		const auto clipOdd = 0;

#define normalizeVertex(v) v.w=1/v.w; v.x*=v.w; v.y*=v.w; v.z*=v.w;v.x=v.x*0.5f+0.5f;v.y=v.y*0.5f+0.5f;
		normalizeVertex(v1);
		normalizeVertex(v2);
		normalizeVertex(v3);
		for (int i = 3; i < retdTriCnt; i++) {
			normalizeVertex(retd[i + retOffset - 3].pos);
		}
#undef normalizeVertex


		// Atomic Insertions
		auto threadId = threadIdx.x;
		const auto frameHeight = csFrameHeight;
		const auto frameWidth = csFrameWidth;
		unsigned idxSrc;
		const auto invFrameHeight = csFrameHeightInv;
		const auto invFrameWidth = csFrameWidthInv;
		for (int i = 0; i < retCnt[clipOdd] - 2; i++) {
			int temp;
#define getBary(fx,fy) (temp = retdIndex[(fx)* possibleTris+(fy)+ retIdxOffset], (temp==0)?ifloat3{1,0,0}:((temp==1)?ifloat3{0,1,0}:(temp==2)?ifloat3{0,0,1}:retd[temp - 3 + retOffset].barycenter))
#define getPos(fx,fy) (temp = retdIndex[(fx)* possibleTris+(fy)+ retIdxOffset], (temp==0)?v1:((temp==1)?v2:(temp==2)?v3:retd[temp - 3 + retOffset].pos))
			AssembledTriangleProposalCUDA atri;
			atri.b1 = getBary(clipOdd, 0);
			atri.b2 = getBary(clipOdd, i+1);
			atri.b3 = getBary(clipOdd, i+2);

			const auto dv1 = getPos(clipOdd, 0);
			const auto dv2 = getPos(clipOdd, i + 1);
			const auto dv3 = getPos(clipOdd, i + 2);
#undef getBary
#undef getPos
			
			if (devViewSpaceClip(dv1, dv2, dv3)) {
				continue;
			}
			if (!devTriangleCull(dv1, dv2, dv3)) {
				continue;
			}
			atri.v1 = dv1.z;
			atri.v2 = dv2.z;
			atri.v3 = dv3.z;

			const float ar = 1.0f / devEdgeFunction(dv1, dv2, dv3);
			const float sV2V1y = dv2.y - dv1.y;
			const float sV2V1x = dv1.x - dv2.x;
			const float sV3V2y = dv3.y - dv2.y;
			const float sV3V2x = dv2.x - dv3.x;
			const float sV1V3y = dv1.y - dv3.y;
			const float sV1V3x = dv3.x - dv1.x;

			atri.f3 = { (float)(sV2V1y * ar) * invFrameWidth, (float)(sV2V1x * ar) * invFrameHeight,(float)((-dv1.x * sV2V1y - dv1.y * sV2V1x) * ar) };
			atri.f1 = { (float)(sV3V2y * ar) * invFrameWidth, (float)(sV3V2x * ar) * invFrameHeight,(float)((-dv2.x * sV3V2y - dv2.y * sV3V2x) * ar) };
			atri.f2 = { (float)(sV1V3y * ar) * invFrameWidth, (float)(sV1V3x * ar) * invFrameHeight,(float)((-dv3.x * sV1V3y - dv3.y * sV1V3x) * ar) };

			const auto dEps = CU_EPS*frameHeight*frameWidth;
			ifloat3 edgeCoefs[3];
			atri.e1 = { (float)(sV2V1y)*frameHeight,  (float)(sV2V1x)*frameWidth ,  (float)(-dv2.x * dv1.y + dv1.x * dv2.y) * frameHeight * frameWidth };
			atri.e2 = { (float)(sV3V2y)*frameHeight,  (float)(sV3V2x)*frameWidth ,  (float)(-dv3.x * dv2.y + dv2.x * dv3.y) * frameHeight * frameWidth };
			atri.e3 = { (float)(sV1V3y)*frameHeight,  (float)(sV1V3x)*frameWidth ,  (float)(-dv1.x * dv3.y + dv3.x * dv1.y) * frameHeight * frameWidth };

			atri.e1.z += dEps;
			atri.e2.z += dEps;
			atri.e3.z += dEps;


			atri.v1 = dv1.z * atri.f1.x + dv2.z * atri.f2.x + dv3.z * atri.f3.x;
			atri.v2 = dv1.z * atri.f1.y + dv2.z * atri.f2.y + dv3.z * atri.f3.y;
			atri.v3 = dv1.z * atri.f1.z + dv2.z * atri.f2.z + dv3.z * atri.f3.z;
			atri.f1.x *= dv1.w;
			atri.f1.y *= dv1.w;
			atri.f1.z *= dv1.w;

			atri.f2.x *= dv2.w;
			atri.f2.y *= dv2.w;
			atri.f2.z *= dv2.w;
			
			atri.f3.x *= dv3.w;
			atri.f3.y *= dv3.w;
			atri.f3.z *= dv3.w;
			atri.originalPrimitive = primId;
			irect2Df bbox;

			auto dAlignedATM2 = static_cast<AssembledTriangleProposalCUDA*>(__builtin_assume_aligned(dAssembledTriangleM2, 128));
			auto curIdx = atomicAdd(&dAssembledTriangleCounterM2, 1);
			dAlignedATM2[curIdx] = atri;
			devGetBBox(dv1, dv2, dv3, bbox);
			dBoundingBoxM2[curIdx] = bbox;
		}
#undef ret
	}

	IFRIT_KERNEL void firstBinnerRasterizerKernel(uint32_t startOffset, uint32_t bound) {
		auto globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
		if (globalInvo >= bound)return;

		//TODO: reduce global memory access
		auto bbox = dBoundingBoxM2[startOffset+globalInvo];

		auto atri = dAssembledTriangleM2[startOffset+globalInvo];
		float bboxSz = min(bbox.w - bbox.x, bbox.h - bbox.y);

		//TODO: reduce branch resolving
		if (bboxSz > CU_LARGE_TRIANGLE_THRESHOLD) {
			devExecuteBinnerLargeTile(startOffset+globalInvo, atri, bbox);
		}
		else {
			devExecuteBinner(startOffset+globalInvo, atri, bbox);
		}

	}

	IFRIT_KERNEL void firstBinnerRasterizerEntryKernel(uint32_t firstTriangle,uint32_t triangleNum) {
		auto totalTriangles = triangleNum;
		auto dispatchBlocks = totalTriangles / CU_FIRST_RASTERIZATION_THREADS + (totalTriangles % CU_FIRST_RASTERIZATION_THREADS != 0);
		firstBinnerRasterizerKernel CU_KARG2(dim3(dispatchBlocks,1,1), dim3(CU_FIRST_RASTERIZATION_THREADS,CU_FIRST_BINNER_STRIDE, CU_FIRST_BINNER_STRIDE)) (
			firstTriangle, totalTriangles
		);
	}

	IFRIT_KERNEL void secondBinnerWorklistRasterizerKernel(int totalElements) {
		const auto globalInvo = (blockIdx.x << 2) + threadIdx.z;
		if (globalInvo > totalElements)return;
		int binId = dRasterQueueWorklistTile[globalInvo];
		int prim = dRasterQueueWorklistPrim[globalInvo];
		int binIdxY = binId / CU_BIN_SIZE;
		int binIdxX = binId % CU_BIN_SIZE;
		int tileIdxX = binIdxX * CU_TILES_PER_BIN + threadIdx.x % CU_TILES_PER_BIN;
		int tileIdxY = binIdxY * CU_TILES_PER_BIN + threadIdx.x / CU_TILES_PER_BIN;

		IFRIT_SHARED ifloat3 e1[128];
		IFRIT_SHARED ifloat3 e2[128];
		IFRIT_SHARED ifloat3 e3[128];


		const auto actGlobalInvo = threadIdx.x + blockDim.x * threadIdx.y + blockDim.y * blockDim.x * threadIdx.z;
		const auto warpId = actGlobalInvo / CU_WARP_SIZE;
		const auto inWarpTid = actGlobalInvo % CU_WARP_SIZE;
		if (inWarpTid == 0) {
			e1[warpId] = dAssembledTriangleM2[prim].e1;
			e2[warpId] = dAssembledTriangleM2[prim].e2;
			e3[warpId] = dAssembledTriangleM2[prim].e3;

		}
		__syncwarp();
		devTilingRasterizationChildProcess(tileIdxX, tileIdxY, prim, threadIdx.y, e1[warpId],e2[warpId],e3[warpId]);
	}

	IFRIT_KERNEL void secondBinnerWorklistRasterizerEntryKernel() {
		const auto totalElements = dRasterQueueWorklistCounter;
		const auto dispatchBlocks = (totalElements / 4) + (totalElements % 4 != 0);
		if (totalElements == 0)return;
		if constexpr (CU_PROFILER_SECOND_BINNER_WORKQUEUE) {
			printf("Second Binner Work Queue Size: %d\n", totalElements);
			printf("Dispatched Thread Blocks %d\n", dispatchBlocks);
		}
		//secondBinnerWorklistRasterizerKernel CU_KARG2(dim3(dispatchBlocks, CU_TILES_PER_BIN* CU_TILES_PER_BIN, 16), dim3(CU_EXPERIMENTAL_SECOND_BINNER_WORKLIST_THREADS, 1, 1)) (totalElements);
		secondBinnerWorklistRasterizerKernel CU_KARG2(dim3(dispatchBlocks, 1, 1), dim3(CU_TILES_PER_BIN * CU_TILES_PER_BIN, 16, 4)) (totalElements);
	}

	IFRIT_HOST void secondBinnerWorklistRasterizerEntryProfileKernel() {
		auto totalElements = 0;
		cudaDeviceSynchronize();
		cudaMemcpyFromSymbol(&totalElements, dRasterQueueWorklistCounter, sizeof(int));
		const auto dispatchBlocks = (totalElements / 4) + (totalElements % 4 != 0);
		if (totalElements == 0)return;
		if constexpr (CU_PROFILER_SECOND_BINNER_WORKQUEUE) {
			printf("Second Binner Work Queue Size: %d\n", totalElements);
			printf("Dispatched Thread Blocks %d\n", dispatchBlocks);
		}
		
		secondBinnerWorklistRasterizerKernel CU_KARG2(dim3(dispatchBlocks, 1, 1), dim3(CU_TILES_PER_BIN * CU_TILES_PER_BIN, 16, 4)) (totalElements);
	}

	IFRIT_KERNEL void fragmentShadingKernel(
		FragmentShader*  fragmentShader,
		int* IFRIT_RESTRICT_CUDA dIndexBuffer,
		const VaryingStore* const* IFRIT_RESTRICT_CUDA dVaryingBuffer,
		ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
		float* IFRIT_RESTRICT_CUDA dDepthBuffer
	) {
		uint32_t tileX = blockIdx.x;
		uint32_t tileY = blockIdx.y;
		uint32_t binX = tileX / CU_TILES_PER_BIN;
		uint32_t binY = tileY / CU_TILES_PER_BIN;

		uint32_t superTileX = tileX / (CU_BINS_PER_LARGE_BIN * CU_TILES_PER_BIN);
		uint32_t superTileY = tileY / (CU_BINS_PER_LARGE_BIN * CU_TILES_PER_BIN);

		uint32_t tileId = tileY * CU_TILE_SIZE + tileX;
		uint32_t binId = binY * CU_BIN_SIZE + binX;
		uint32_t superTileId = superTileY * CU_LARGE_BIN_SIZE + superTileX;

		const auto frameWidth = csFrameWidth;
		const auto frameHeight = csFrameHeight;
		
		const auto completeCandidates = dCoverQueueFullM2[binId].getSize();
		const auto largeCandidates = dCoverQueueSuperTileFullM2[superTileId].getSize();

		constexpr auto vertexStride = CU_TRIANGLE_STRIDE;
		const auto varyingCount = csVaryingCounts;

		const int threadX = threadIdx.x;
		const int threadY = threadIdx.y;
		const int blockX = blockDim.x;
		const int blockY = blockDim.y;
		const int bds = blockDim.x * blockDim.y;
		const auto threadId = threadY * blockDim.x + threadX;

		const int pixelXS = threadX + tileX * csFrameWidth / CU_TILE_SIZE;
		const int pixelYS = threadY + tileY * csFrameHeight / CU_TILE_SIZE;

		float localDepthBuffer = 1;
		float candidateBary[3];
		int candidatePrim = -1;
		const float compareDepth = dDepthBuffer[pixelYS * frameWidth + pixelXS];
		float pDx = 1.0f * pixelXS;
		float pDy = 1.0f * pixelYS;

		IFRIT_SHARED float colorOutputSingle[256 * 4];
		IFRIT_SHARED float interpolatedVaryings[256 * 4 * CU_MAX_VARYINGS];
		auto shadingPass = [&](const AssembledTriangleProposalCUDA& atp) {
			candidateBary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z);
			candidateBary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z);
			candidateBary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z);
			float zCorr = 1.0f / (candidateBary[0] + candidateBary[1] + candidateBary[2]);
			candidateBary[0] *= zCorr;
			candidateBary[1] *= zCorr;
			candidateBary[2] *= zCorr;

			
			float desiredBary[3];
			desiredBary[0] = candidateBary[0] * atp.b1.x + candidateBary[1] * atp.b2.x + candidateBary[2] * atp.b3.x;
			desiredBary[1] = candidateBary[0] * atp.b1.y + candidateBary[1] * atp.b2.y + candidateBary[2] * atp.b3.y;
			desiredBary[2] = candidateBary[0] * atp.b1.z + candidateBary[1] * atp.b2.z + candidateBary[2] * atp.b3.z;
			auto addr = dIndexBuffer + atp.originalPrimitive * vertexStride;
			for (int k = 0; k < varyingCount; k++) {
				const auto va = dVaryingBuffer[k];
				VaryingStore vd;
				vd.vf4 = { 0,0,0,0 };
				for (int j = 0; j < 3; j++) {
					auto vaf4 = va[addr[j]].vf4;
					vd.vf4.x += vaf4.x * desiredBary[j];
					vd.vf4.y += vaf4.y * desiredBary[j];
					vd.vf4.z += vaf4.z * desiredBary[j];
					vd.vf4.w += vaf4.w * desiredBary[j];
				}
				auto dest = (ifloat4s256*)(interpolatedVaryings + 1024 * k + threadId);
				dest->x = vd.vf4.x;
				dest->y = vd.vf4.y;
				dest->z = vd.vf4.z;
				dest->w = vd.vf4.w;
			}
			fragmentShader->execute(interpolatedVaryings + threadId, colorOutputSingle + threadId, 1);

			auto col0 = static_cast<ifloat4*>(__builtin_assume_aligned(dColorBuffer[0], 16));
			ifloat4 finalRgba;
			ifloat4s256 midOutput = ((ifloat4s256*)(colorOutputSingle + threadId))[0];
			finalRgba.x = midOutput.x;
			finalRgba.y = midOutput.y;
			finalRgba.z = midOutput.z;
			finalRgba.w = midOutput.w;

			col0[pixelYS * frameWidth + pixelXS] = finalRgba;
			if constexpr (CU_PROFILER_OVERDRAW) {
				atomicAdd(&dOverDrawCounter, 1);
			}
		};

		auto zPrePass = [&](const AssembledTriangleProposalCUDA& atp,  int primId) {
			float bary[3];
			float interpolatedDepth;
			if constexpr (CU_OPT_COMPRESSED_Z_INTERPOL) {
				interpolatedDepth = atp.v1 * pDx + atp.v2 * pDy + atp.v3;
			}
			else {
				bary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z);
				bary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z);
				bary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z);
				interpolatedDepth = bary[0] * atp.v1 + bary[1] * atp.v2 + bary[2] * atp.v3;
			}
			if constexpr (CU_OPT_EXPERIMENTAL_PERFORMANCE) {
				const auto fp = interpolatedDepth - localDepthBuffer;
				int cmpv = signbit(fp);
				localDepthBuffer += cmpv * fp;
				candidatePrim += cmpv * (primId - candidatePrim);
			}
			else {
				if (interpolatedDepth < localDepthBuffer) {
					localDepthBuffer = interpolatedDepth;
					candidatePrim = primId;
				}
			}
			if constexpr (CU_PROFILER_OVERDRAW) {
				atomicAdd(&dOverZTestCounter, 1);
			}
		};

		// Large Bin Level
		if constexpr (true) {
			int log2LargeCandidates = 31 - __clz(max(0, largeCandidates - 1) | ((1 << CU_VECTOR_BASE_LENGTH) - 1)) - (CU_VECTOR_BASE_LENGTH - 1);
			for (int i = 0; i < log2LargeCandidates; i++) {
				int dmax = i ? (1 << (i + CU_VECTOR_BASE_LENGTH - 1)) : (1 << (CU_VECTOR_BASE_LENGTH));
				const auto& data = dCoverQueueSuperTileFullM2[superTileId].data[i];
				for (int j = 0; j < dmax; j++) {
					const auto proposal = data[j];
					const auto atp = dAssembledTriangleM2[proposal];
					zPrePass(atp, proposal);
				}
			}
			int processedLargeProposals = log2LargeCandidates ? (1 << (log2LargeCandidates + CU_VECTOR_BASE_LENGTH - 1)) : 0;
			const auto& dataPixel = dCoverQueueSuperTileFullM2[superTileId].data[log2LargeCandidates];
			const auto limPixel = largeCandidates - log2LargeCandidates;
			for (int i = 0; i < limPixel; i++) {
				const auto proposal = dataPixel[i];
				const auto atp = dAssembledTriangleM2[proposal];
				zPrePass(atp, proposal);
			}
		}


		// Bin Level
		for (int i = completeCandidates - 1; i >= 0; i--) {
			const auto proposal = dCoverQueueFullM2[binId].at(i);
			const auto atp = dAssembledTriangleM2[proposal];
			zPrePass(atp, proposal);
		}
		
		// Pixel Level
		int sbId;
		if constexpr (true) {
			auto curTileX = (tileX + 0) * frameWidth / CU_TILE_SIZE;
			auto curTileY = (tileY + 0) * frameHeight / CU_TILE_SIZE;
			auto curTileX2 = (tileX + 1) * frameWidth / CU_TILE_SIZE;
			auto curTileY2 = (tileY + 1) * frameHeight / CU_TILE_SIZE;
			auto curTileWid = curTileX2 - curTileX;
			auto curTileHei = curTileY2 - curTileY;
			int numSubtilesX = curTileWid / CU_EXPERIMENTAL_SUBTILE_WIDTH + (curTileWid % CU_EXPERIMENTAL_SUBTILE_WIDTH != 0);
			int numSubtilesY = curTileHei / CU_EXPERIMENTAL_SUBTILE_WIDTH + (curTileHei % CU_EXPERIMENTAL_SUBTILE_WIDTH != 0);
			int inTileX = pixelXS - tileX * frameWidth / CU_TILE_SIZE;
			int inTileY = pixelYS - tileY * frameHeight / CU_TILE_SIZE;
			int inSubTileX = inTileX / CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int inSubTileY = inTileY / CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int inSubTileId = inSubTileY * numSubtilesX + inSubTileX;
			sbId = inSubTileId;
			const int pixelCandidates = dCoverQueuePixelM2[tileId][inSubTileId].getSize();
			int log2PixelCandidates = 31 - __clz(max(0, pixelCandidates - 1) | ((1 << CU_VECTOR_BASE_LENGTH) - 1)) - (CU_VECTOR_BASE_LENGTH - 1);
			int dwX = inTileX % CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int dwY = inTileY % CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int dwId = dwY * CU_EXPERIMENTAL_SUBTILE_WIDTH + dwX;
			int dwMask = (1 << dwId);

			for (int i = 0; i < log2PixelCandidates; i++) {
				int dmax = i ? (1 << (i + CU_VECTOR_BASE_LENGTH - 1)) : (1 << (CU_VECTOR_BASE_LENGTH));
				const auto& data = dCoverQueuePixelM2[tileId][inSubTileId].data[i];
				for (int j = 0; j < dmax; j++) {
					const auto proposal = data[j];
					const auto atp = dAssembledTriangleM2[proposal.primId];
					if ((proposal.mask & dwMask)) {
						zPrePass(atp, proposal.primId);
					}
					else {
						if constexpr (CU_PROFILER_OVERDRAW) {
							atomicAdd(&dIrrelevantPixel, 1);
						}
					}
					if constexpr (CU_PROFILER_OVERDRAW) {
						atomicAdd(&dTotalBinnedPixel, 1);
					}
				}
			}
			int processedPixelProposals = log2PixelCandidates ? (1 << (log2PixelCandidates + CU_VECTOR_BASE_LENGTH - 1)) : 0;
			const auto& dataPixel = dCoverQueuePixelM2[tileId][inSubTileId].data[log2PixelCandidates];
			const auto limPixel = pixelCandidates - processedPixelProposals;
			for (int i = 0; i < limPixel; i++) {
				const auto proposal = dataPixel[i];
				const auto atp = dAssembledTriangleM2[proposal.primId];
				if ((proposal.mask & dwMask)) {
					zPrePass(atp, proposal.primId);
				}
				else {
					if constexpr (CU_PROFILER_OVERDRAW) {
						atomicAdd(&dIrrelevantPixel, 1);
					}
				}
				if constexpr (CU_PROFILER_OVERDRAW) {
					atomicAdd(&dTotalBinnedPixel, 1);
				}
			}
		
		}

		
		if (candidatePrim != -1 && localDepthBuffer< compareDepth) {
			shadingPass(dAssembledTriangleM2[candidatePrim]);
			dDepthBuffer[pixelYS * frameWidth + pixelXS] = localDepthBuffer;
		}

	}
	IFRIT_KERNEL void updateHierZBuffer(float* depthBuffer) {
		const auto tileIdX = blockIdx.x;
		const auto tileIdY = threadIdx.x;
		const auto tileId = tileIdY * CU_TILE_SIZE + tileIdX;
		
		auto curTileX = tileIdX * csFrameWidth / CU_TILE_SIZE;
		auto curTileY = tileIdY * csFrameHeight / CU_TILE_SIZE;
		auto curTileX2 = (tileIdX + 1) * csFrameWidth / CU_TILE_SIZE;
		auto curTileY2 = (tileIdY + 1) * csFrameHeight / CU_TILE_SIZE;

		auto minv = 0.0f;
		for (int i = curTileX; i < curTileX2; i++) {
			for (int j = curTileY; j < curTileY2; j++) {
				minv = max(minv, depthBuffer[j * csFrameWidth + i]);
			}
		}
		dHierarchicalDepthTile[tileId] = minv;
	}

	IFRIT_KERNEL void integratedResetKernel() {
		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
		dAssembledTriangleCounterM2 = 0;
		dRasterQueueWorklistCounter = 0;
		for (int i = 0; i < CU_MAX_SUBTILES_PER_TILE; i++) {
			dCoverQueuePixelM2[globalInvocation][i].clear();
		}
		if (globalInvocation < CU_LARGE_BIN_SIZE * CU_LARGE_BIN_SIZE) {
			dCoverQueueSuperTileFullM2[globalInvocation].clear();
		}
		if (globalInvocation < CU_BIN_SIZE * CU_BIN_SIZE) {
			dCoverQueueFullM2[globalInvocation].clear();
			dRasterQueueM2[globalInvocation].clear();
		}
		if constexpr (CU_PROFILER_OVERDRAW) {
			if (blockIdx.x == 0 && threadIdx.x == 0) {
				printf("Overdraw Rate:%f (%d)\n", dOverDrawCounter * 1.0f / 2048.0f / 2048.0f, dOverDrawCounter);
				printf("ZTest Rate:%f (%d)\n", dOverZTestCounter * 1.0f / 2048.0f / 2048.0f , dOverZTestCounter);
				printf("Irrelevant Rate:%f (%d/%d)\n", 1.0f * dIrrelevantPixel / dTotalBinnedPixel, dIrrelevantPixel, dTotalBinnedPixel);

				dOverDrawCounter = 0;
				dOverZTestCounter = 0;
				dIrrelevantPixel = 0;
				dTotalBinnedPixel = 0;
			}
		}
	}

	IFRIT_KERNEL void integratedInitKernel() {
		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
		dAssembledTriangleCounterM2 = 0;
		dSecondBinnerActiveReqs = 0;
		dSecondBinnerTotalReqs = 0;
		dSecondBinnerEmptyTiles = 0;
		dRasterQueueWorklistCounter = 0;
		for (int i = 0; i < CU_MAX_SUBTILES_PER_TILE; i++) {
			dCoverQueuePixelM2[globalInvocation][i].initialize();
		}
		if (globalInvocation < CU_LARGE_BIN_SIZE * CU_LARGE_BIN_SIZE) {
			dCoverQueueSuperTileFullM2[globalInvocation].initialize();
		}
		if (globalInvocation < CU_BIN_SIZE * CU_BIN_SIZE) {
			dCoverQueueFullM2[globalInvocation].initialize();
			dRasterQueueM2[globalInvocation].initialize();
		}
	}

	IFRIT_KERNEL void resetLargeTileKernel(bool resetTriangle) {
		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
		for (int i = 0; i < CU_MAX_SUBTILES_PER_TILE; i++) {
			dCoverQueuePixelM2[globalInvocation][i].clear();
		}
		if (globalInvocation < CU_LARGE_BIN_SIZE * CU_LARGE_BIN_SIZE) {
			dCoverQueueSuperTileFullM2[globalInvocation].clear();
		}
		if (globalInvocation < CU_BIN_SIZE * CU_BIN_SIZE) {
			dCoverQueueFullM2[globalInvocation].clear();
			dRasterQueueM2[globalInvocation].clear();
		}

		if constexpr (CU_PROFILER_SECOND_BINNER_UTILIZATION) {
			if (globalInvocation == 0) {
				printf("Tile Empty Rate: %f (%d/%d)\n",
					1.0f * dSecondBinnerEmptyTiles / (CU_TILE_SIZE * CU_TILE_SIZE), dSecondBinnerEmptyTiles, CU_TILE_SIZE * CU_TILE_SIZE);
				printf("Second Binner Actual Utilization: %f (%d/%d)\n", 
					1.0f * dSecondBinnerActiveReqs / dSecondBinnerTotalReqs, dSecondBinnerActiveReqs, dSecondBinnerTotalReqs);
				dSecondBinnerActiveReqs = 0;
				dSecondBinnerTotalReqs = 0;
				dSecondBinnerEmptyTiles = 0;
			}
		}

		if (globalInvocation == 0) {
			if (resetTriangle) {
				dAssembledTriangleCounterM2 = 0;
				if constexpr (CU_PROFILER_TRIANGLE_SETUP) {
					printf("Primitive Queue Occupation: %f (%d/%d)\n",
						1.0f * dAssembledTriangleCounterM2 / (CU_SINGLE_TIME_TRIANGLE * 2), dAssembledTriangleCounterM2, CU_SINGLE_TIME_TRIANGLE * 2);
				}
			}
			dRasterQueueWorklistCounter = 0;
		}
	}

	IFRIT_KERNEL void imageResetFloat32MonoKernel(
		float* dBuffer,
		float value
	) {
		const auto invoX = blockIdx.x * blockDim.x + threadIdx.x;
		const auto invoY = blockIdx.y * blockDim.y + threadIdx.y;
		if (invoX >= csFrameWidth || invoY >= csFrameHeight) {
			return;
		}
		dBuffer[(invoY * csFrameWidth + invoX)] = value;
	}

	IFRIT_KERNEL void imageResetFloat32RGBAKernel(
		float* dBuffer,
		float value
	) {
		const auto invoX = blockIdx.x * blockDim.x + threadIdx.x;
		const auto invoY = blockIdx.y * blockDim.y + threadIdx.y;
		if (invoX >= csFrameWidth || invoY >= csFrameHeight) {
			return;
		}
		const auto dAlignedBuffer = static_cast<float*>(__builtin_assume_aligned(dBuffer, 16));
		dBuffer[(invoY * csFrameWidth + invoX) * 4 + 0] = value;
		dBuffer[(invoY * csFrameWidth + invoX) * 4 + 1] = value;
		dBuffer[(invoY * csFrameWidth + invoX) * 4 + 2] = value;
		dBuffer[(invoY * csFrameWidth + invoX) * 4 + 3] = value;
	}

	IFRIT_KERNEL void unifiedRasterEngineStageIIKernel(
		int* dIndexBuffer,
		const VaryingStore const* const* dVaryingBuffer,
		ifloat4** dColorBuffer,
		float* dDepthBuffer,
		FragmentShader* dFragmentShader,
		int tileSizeX,
		int tileSizeY,
		bool isTailCall
	) {
		if constexpr (CU_OPT_II_SKIP_ON_FEW_GEOMETRIES) {
			if (!isTailCall && dAssembledTriangleCounterM2 < CU_EXPERIMENTAL_II_FEW_GEOMETRIES_LIMIT) {
				return;
			}
		}
		else {
			if constexpr (CU_OPT_II_SKIP_ON_EMPTY_GEOMETRY) {
				if (dAssembledTriangleCounterM2 == 0) {
					return;
				}
			}
		}
		int totalTms = dAssembledTriangleCounterM2 / CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;
		int curTime = -1;
		for (int sI = 0; sI < dAssembledTriangleCounterM2; sI += CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER) {
			curTime++;
			int length = min(dAssembledTriangleCounterM2 - sI, CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER);
			int start = sI;
			bool isLast = curTime == totalTms;
			if (curTime == totalTms - 1) {
				if (dAssembledTriangleCounterM2 % CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER < CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER * 2 / 5) {
					length = dAssembledTriangleCounterM2 - sI;
					sI += CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;
					isLast = true;
				}
			}
			Impl::firstBinnerRasterizerEntryKernel CU_KARG2(1, 1)(start, length);
			Impl::secondBinnerWorklistRasterizerEntryKernel CU_KARG2(1, 1)();
			Impl::fragmentShadingKernel CU_KARG2(dim3(CU_TILE_SIZE, CU_TILE_SIZE, 1), dim3(tileSizeX, tileSizeY, 1)) (
				dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer
				);
			Impl::resetLargeTileKernel CU_KARG2(CU_TILE_SIZE, CU_TILE_SIZE)(isLast);
		}
	}
	

	IFRIT_KERNEL void unifiedRasterEngineKernel(
		int totalIndices,
		ifloat4* dPositionBuffer,
		int* dIndexBuffer,
		const VaryingStore* const* dVaryingBuffer,
		ifloat4** dColorBuffer,
		float* dDepthBuffer,
		FragmentShader* dFragmentShader,
		int tileSizeX,
		int tileSizeY
	) {
		for (int i = 0; i < totalIndices; i += CU_SINGLE_TIME_TRIANGLE * 3) {
			auto indexCount = min((int)(CU_SINGLE_TIME_TRIANGLE * 3), totalIndices - i);
			bool isTailCall = (i + CU_SINGLE_TIME_TRIANGLE * 3) >= totalIndices;
			int geometryExecutionBlocks = (indexCount / CU_TRIANGLE_STRIDE / CU_GEOMETRY_PROCESSING_THREADS) + ((indexCount / CU_TRIANGLE_STRIDE % CU_GEOMETRY_PROCESSING_THREADS) != 0);
			
			Impl::geometryProcessingKernel CU_KARG2(geometryExecutionBlocks, CU_GEOMETRY_PROCESSING_THREADS)(
				dPositionBuffer, dIndexBuffer, i, indexCount);
			unifiedRasterEngineStageIIKernel CU_KARG2(1, 1)(
				dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer, dFragmentShader, tileSizeX, tileSizeY, isTailCall
			);
			
		}
	}

	IFRIT_HOST void unifiedRasterEngineProfileEntry(
		int totalIndices,
		ifloat4* dPositionBuffer,
		int* dIndexBuffer,
		const VaryingStore* const* dVaryingBuffer,
		ifloat4** dColorBuffer,
		float* dDepthBuffer,
		FragmentShader* dFragmentShader,
		int tileSizeX,
		int tileSizeY,
		cudaStream_t& compStream
	) {
		for (int i = 0; i < totalIndices; i += CU_SINGLE_TIME_TRIANGLE * 3) {
			auto indexCount = min((int)(CU_SINGLE_TIME_TRIANGLE * 3), totalIndices - i);
			bool isTailCall = (i + CU_SINGLE_TIME_TRIANGLE * 3) >= totalIndices;
			int geometryExecutionBlocks = (indexCount / CU_TRIANGLE_STRIDE / CU_GEOMETRY_PROCESSING_THREADS) + ((indexCount / CU_TRIANGLE_STRIDE % CU_GEOMETRY_PROCESSING_THREADS) != 0);
			Impl::geometryProcessingKernel CU_KARG2(geometryExecutionBlocks, CU_GEOMETRY_PROCESSING_THREADS)(
				dPositionBuffer, dIndexBuffer, i, indexCount);
			uint32_t dAssembledTriangleCounterM2Host = 0;
			cudaDeviceSynchronize();
			cudaMemcpyFromSymbol(&dAssembledTriangleCounterM2Host, dAssembledTriangleCounterM2, sizeof(uint32_t));
			if constexpr (CU_OPT_II_SKIP_ON_FEW_GEOMETRIES) {
				if (!isTailCall && dAssembledTriangleCounterM2Host < CU_EXPERIMENTAL_II_FEW_GEOMETRIES_LIMIT) {
					continue;
				}
			}
			else {
				if constexpr (CU_OPT_II_SKIP_ON_EMPTY_GEOMETRY) {
					if (dAssembledTriangleCounterM2Host == 0) {
						continue;
					}
				}
			}
			int totalTms = dAssembledTriangleCounterM2Host / CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;
			int curTime = -1;
			for (int sI = 0; sI < dAssembledTriangleCounterM2Host; sI += CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER) {
				curTime++;
				int length = min(dAssembledTriangleCounterM2Host - sI, CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER);
				int start = sI;
				bool isLast = (curTime == totalTms);
				if (curTime == totalTms - 1) {
					if (dAssembledTriangleCounterM2 % CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER < CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER * 2 / 5) {
						length = dAssembledTriangleCounterM2Host - sI;
						sI += CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;
						isLast = true;
					}
				}
				Impl::firstBinnerRasterizerEntryKernel CU_KARG4(1, 1, 0, compStream)(start,length);
				Impl::secondBinnerWorklistRasterizerEntryProfileKernel();
				Impl::fragmentShadingKernel CU_KARG4(dim3(CU_TILE_SIZE, CU_TILE_SIZE, 1), dim3(tileSizeX, tileSizeY, 1), 0, compStream) (
					dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer
					);
				Impl::resetLargeTileKernel CU_KARG4(CU_TILE_SIZE, CU_TILE_SIZE, 0, compStream)(isLast);
			}
		}
		cudaDeviceSynchronize();
	}
}


namespace  Ifrit::Engine::TileRaster::CUDA::Invocation {


	template<typename T>
	__global__ void kernFixVTable(T* devicePtr) {
		T temp(*devicePtr);
		memcpy(devicePtr, &temp, sizeof(T));
	}

	template<typename T>
	__host__ T* hostGetDeviceObjectCopy(T* hostObject) {
		T* deviceHandle;
		cudaMalloc(&deviceHandle, sizeof(T));
		cudaMemcpy(deviceHandle, hostObject, sizeof(T), cudaMemcpyHostToDevice);
		printf("Copying object to CUDA, %lld,%d\n", deviceHandle, 1);
		kernFixVTable<T> CU_KARG2(1, 1)(deviceHandle);
		cudaDeviceSynchronize();
		printf("Cuda Addr=%lld\n", deviceHandle);
		printf("Object copied to CUDA\n");
		return deviceHandle;
	}

	template<class T>
	T* copyShaderToDevice(T* x) {
		return hostGetDeviceObjectCopy<T>(x);
	}

	char* deviceMalloc(uint32_t size) {
		char* ptr;
		cudaMalloc(&ptr, size);
		return ptr;
	
	}
	void deviceFree(char* ptr) {
		cudaFree(ptr);
	}


	int* getIndexBufferDeviceAddr(const int* hIndexBuffer, uint32_t indexBufferSize, int* dOldIndexBuffer) {
		if(dOldIndexBuffer != nullptr) {
			cudaFree(dOldIndexBuffer);
		}
		int* dIndexBuffer;
		cudaMalloc(&dIndexBuffer, indexBufferSize * sizeof(int));
		cudaMemcpy(dIndexBuffer, hIndexBuffer, indexBufferSize * sizeof(int), cudaMemcpyHostToDevice);
		return dIndexBuffer;
	}
	char* getVertexBufferDeviceAddr(const char* hVertexBuffer, uint32_t bufferSize, char* dOldBuffer) {
		if(dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		char* dBuffer;
		cudaMalloc(&dBuffer, bufferSize);
		cudaMemcpy(dBuffer, hVertexBuffer, bufferSize, cudaMemcpyHostToDevice);
		return dBuffer;
	
	}
	TypeDescriptorEnum* getTypeDescriptorDeviceAddr(const TypeDescriptorEnum* hBuffer, uint32_t bufferSize, TypeDescriptorEnum* dOldBuffer) {
		if(dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		TypeDescriptorEnum* dBuffer;
		cudaMalloc(&dBuffer, bufferSize * sizeof(TypeDescriptorEnum));
		cudaMemcpy(dBuffer, hBuffer, bufferSize * sizeof(TypeDescriptorEnum), cudaMemcpyHostToDevice);
		return dBuffer;
	
	}
	float* getDepthBufferDeviceAddr(uint32_t bufferSize, float* dOldBuffer) {
		if(dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		float* dBuffer;
		cudaMalloc(&dBuffer, bufferSize * sizeof(float));
		return dBuffer;
	}
	ifloat4* getPositionBufferDeviceAddr(uint32_t bufferSize, ifloat4* dOldBuffer) {
		if(dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		ifloat4* dBuffer;
		cudaMalloc(&dBuffer, bufferSize * sizeof(ifloat4));
		return dBuffer;
	
	}

	void getColorBufferDeviceAddr(
		const std::vector<ifloat4*>& hColorBuffer,
		std::vector<ifloat4*>& dhColorBuffer,
		ifloat4**& dColorBuffer,
		uint32_t bufferSize,
		std::vector<ifloat4*>& dhOldColorBuffer,
		ifloat4** dOldBuffer
	) {
		//TODO: POTENTIAL BUGS & CUDA DEVICE MEMORY LEAK
		if (dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		dhColorBuffer.resize(hColorBuffer.size());

		cudaMalloc(&dColorBuffer, hColorBuffer.size() * sizeof(ifloat4*));
		for (int i = 0; i < hColorBuffer.size(); i++) {
			cudaMalloc(&dhColorBuffer[i], bufferSize * sizeof(ifloat4));
			cudaMemcpy(dhColorBuffer[i], hColorBuffer[i], bufferSize * sizeof(ifloat4), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(dColorBuffer, dhColorBuffer.data(), hColorBuffer.size() * sizeof(ifloat4*), cudaMemcpyHostToDevice);
	

	}
	void updateFrameBufferConstants(uint32_t width,uint32_t height) {
		auto invWidth = 1.0f / width;
		auto invHeight = 1.0f / height;
		cudaMemcpyToSymbol(Impl::csFrameWidthInv, &invWidth, sizeof(float));
		cudaMemcpyToSymbol(Impl::csFrameHeightInv, &invHeight, sizeof(float));
		cudaMemcpyToSymbol(Impl::csFrameWidth, &width, sizeof(uint32_t));
		cudaMemcpyToSymbol(Impl::csFrameHeight, &height, sizeof(uint32_t));
		Impl::hsFrameHeight = height;
		Impl::hsFrameWidth = width;
	}


	void updateVarying(uint32_t varyingCounts) {
		Impl::hsVaryingCounts = varyingCounts;
		cudaMemcpyToSymbol(Impl::csVaryingCounts, &varyingCounts, sizeof(uint32_t));
	}

	void updateAttributes(uint32_t attributeCounts) {
		Impl::hsAttributeCounts = attributeCounts;
		cudaMemcpyToSymbol(Impl::csAttributeCounts, &attributeCounts, sizeof(uint32_t));
	}

	void updateVertexCount(uint32_t vertexCount) {
		Impl::hsVertexCounts = vertexCount;
		cudaMemcpyToSymbol(Impl::csVertexCount, &vertexCount, sizeof(uint32_t));
	}

	void initCudaRendering() {
		cudaMemcpyToSymbol(Impl::csCounterClosewiseCull, &Impl::hsCounterClosewiseCull, sizeof(Impl::hsCounterClosewiseCull));
	}

	void updateVertexLayout(TypeDescriptorEnum* dVertexTypeDescriptor, int attrCounts) {
		Impl::hsTotalVertexOffsets = 0;
		for (int i = 0; i < attrCounts; i++) {
			int cof = 0;
			Impl::hsVertexOffsets[i] = Impl::hsTotalVertexOffsets;
			if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT1) cof = sizeof(float);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT2) cof = sizeof(ifloat2);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT3) cof = sizeof(ifloat3);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT4)cof = sizeof(ifloat4);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT1) cof = sizeof(int);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT2) cof = sizeof(iint2);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT3) cof = sizeof(iint3);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT4) cof = sizeof(iint4);
			Impl::hsTotalVertexOffsets += cof;
		}
		cudaMemcpyToSymbol(Impl::csVertexOffsets, Impl::hsVertexOffsets, sizeof(Impl::hsVertexOffsets));
		cudaMemcpyToSymbol(Impl::csTotalVertexOffsets, &Impl::hsTotalVertexOffsets, sizeof(Impl::hsTotalVertexOffsets));
	}

	void invokeCudaRendering(
		char* dVertexBuffer,
		TypeDescriptorEnum* dVertexTypeDescriptor,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		int* dIndexBuffer,
		int* dShaderLockBuffer,
		VertexShader* dVertexShader,
		FragmentShader* dFragmentShader,
		ifloat4** dColorBuffer,
		ifloat4** dHostColorBuffer,
		ifloat4** hColorBuffer,
		uint32_t dHostColorBufferSize,
		float* dDepthBuffer,
		ifloat4* dPositionBuffer,
		TileRasterDeviceContext* deviceContext,
		int totalIndices,
		bool doubleBuffering,
		ifloat4** dLastColorBuffer,
		float aggressiveRatio
	) IFRIT_AP_NOTHROW {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		// Stream Preparation
		static int initFlag = 0;
		static int secondPass = 0;
		static cudaStream_t copyStream, computeStream;
		static cudaEvent_t  copyStart, copyEnd;

		if (secondPass == 1) {
			if constexpr (CU_OPT_CUDA_PROFILE) {
				cudaProfilerStart();
			}
			secondPass = 2;
		}
		if (initFlag == 0) {
			initFlag = 1;
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, CU_HEAP_MEMORY_SIZE);

			cudaStreamCreate(&copyStream);
			cudaStreamCreate(&computeStream);
			cudaEventCreate(&copyStart);
			cudaEventCreate(&copyEnd);

			Impl::integratedInitKernel CU_KARG4(CU_TILE_SIZE, CU_TILE_SIZE,0, computeStream)();
			cudaDeviceSynchronize();
			printf("CUDA Init Done\n");
			
		}
		if (initFlag < 20) {
			initFlag++;
			if (initFlag == 20) {
				secondPass = 1;
			}
		}
		
		// Compute
		std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
		const int tileSizeX = (Impl::hsFrameWidth / CU_TILE_SIZE) + ((Impl::hsFrameWidth % CU_TILE_SIZE) != 0);
		const int tileSizeY = (Impl::hsFrameHeight / CU_TILE_SIZE) + ((Impl::hsFrameHeight % CU_TILE_SIZE) != 0);

		constexpr int dispatchThreadsX = 8;
		constexpr int dispatchThreadsY = 8;
		int dispatchBlocksX = (Impl::hsFrameWidth / dispatchThreadsX) + ((Impl::hsFrameWidth % dispatchThreadsX) != 0);
		int dispatchBlocksY = (Impl::hsFrameHeight / dispatchThreadsY) + ((Impl::hsFrameHeight % dispatchThreadsY) != 0);
		
		Impl::imageResetFloat32MonoKernel CU_KARG4(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY), 0, computeStream)(
			dDepthBuffer, 255.0f
		);
		for (int i = 0; i < dHostColorBufferSize; i++) {
			Impl::imageResetFloat32RGBAKernel CU_KARG4(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY), 0, computeStream)(
				(float*)dHostColorBuffer[i], 0.0f
			);
		}
		int vertexExecutionBlocks = (Impl::hsVertexCounts / CU_VERTEX_PROCESSING_THREADS) + ((Impl::hsVertexCounts % CU_VERTEX_PROCESSING_THREADS) != 0);
		Impl::vertexProcessingKernel CU_KARG4(vertexExecutionBlocks, CU_VERTEX_PROCESSING_THREADS, 0, computeStream)(
			dVertexShader, Impl::hsVertexCounts, dVertexBuffer, dVertexTypeDescriptor,
			deviceContext->dVaryingBuffer, dVaryingTypeDescriptor, dPositionBuffer
		);
		
		constexpr int totalTiles = CU_TILE_SIZE * CU_TILE_SIZE;
		Impl::integratedResetKernel CU_KARG4(CU_TILE_SIZE, CU_TILE_SIZE, 0, computeStream)();


		if constexpr (CU_PROFILER_II_CPU_NSIGHT) {
			Impl::unifiedRasterEngineProfileEntry(
				totalIndices, dPositionBuffer, dIndexBuffer, deviceContext->dVaryingBuffer, dColorBuffer, dDepthBuffer, dFragmentShader,
				tileSizeX, tileSizeY, computeStream
			);
		}
		else {
			Impl::unifiedRasterEngineKernel CU_KARG4(1, 1, 0, computeStream)(
				totalIndices, dPositionBuffer, dIndexBuffer, deviceContext->dVaryingBuffer, dColorBuffer, dDepthBuffer, dFragmentShader,
				tileSizeX, tileSizeY
				);
		}
		
		if (!doubleBuffering) {
			cudaDeviceSynchronize();
		}

		// Memory Copy
		std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
		if (doubleBuffering) {
			for (int i = 0; i < dHostColorBufferSize; i++) {
				//cudaMemcpyAsync(hColorBuffer[i], dLastColorBuffer[i], Impl::hsFrameWidth * Impl::hsFrameHeight * sizeof(ifloat4), cudaMemcpyDeviceToHost, copyStream);
			}
		}
		cudaStreamSynchronize(computeStream);
		if (doubleBuffering) {
			cudaStreamSynchronize(copyStream);
		}

		if (!doubleBuffering) {
			for (int i = 0; i < dHostColorBufferSize; i++) {
				cudaMemcpy(hColorBuffer[i], dHostColorBuffer[i], Impl::csFrameWidth * Impl::csFrameHeight * sizeof(ifloat4), cudaMemcpyDeviceToHost);
			}
		}
		std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();

		// End of rendering
		auto copybackTimes = std::chrono::duration_cast<std::chrono::microseconds>(end3 - end1).count();

		static long long w = 0;
		static long long wt = 0;
		w += copybackTimes;
		wt += 1;
		printf("AvgTime:%lld, FPS=%f, Loops=%d\n", w / wt, 1000000.0f / w * wt, totalIndices / (CU_SINGLE_TIME_TRIANGLE * 3));
		//printf("Memcpy,Compute,Copyback,Counter: %lld,%lld,%lld,%d\n", memcpyTimes, computeTimes, copybackTimes,cntw);
	}
}