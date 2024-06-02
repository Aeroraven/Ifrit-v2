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

	static int hsFrameWidth = 0;
	static int hsFrameHeight = 0;
	static bool hsCounterClosewiseCull = false;
	static int hsVertexOffsets[CU_MAX_ATTRIBUTES];
	static int hsTotalVertexOffsets = 0;

	template<class T>
	class LocalDynamicVector {
	private:
		int baseSize;
		T* data[CU_VECTOR_HIERARCHY_LEVEL];
		int capacity;
		int size;
		int lock;
		int visitors;
	public:
		IFRIT_DEVICE void initialize() {
			baseSize = CU_VECTOR_BASE_LENGTH;
			capacity = (1 << baseSize);
			size = 0;
			data[0] = (T*)malloc(sizeof(T) * capacity);
			data[1] = (T*)malloc(sizeof(T) * capacity * 2);
			lock = 0;
			visitors = 0;
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
	IFRIT_DEVICE static LocalDynamicVector<TileBinProposalCUDA> dCoverQueuePartialM2[CU_TILE_SIZE * CU_TILE_SIZE];

	IFRIT_DEVICE static AssembledTriangleProposalCUDA dAssembledTriangleM2[CU_SINGLE_TIME_TRIANGLE * 2];
	IFRIT_DEVICE static uint32_t dAssembledTriangleCounterM2;

	IFRIT_DEVICE float devEdgeFunction(ifloat4 a, ifloat4 b, ifloat4 c) {
		return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
	}
	IFRIT_DEVICE bool devTriangleCull(ifloat4 v1, ifloat4 v2, ifloat4 v3) {
		float a1 = 1.0f / v1.w;
		float a2 = 1.0f / v2.w;
		float a3 = 1.0f / v3.w;
		float d1 = (v1.x * v2.y) * a1 * a2;
		float d2 = (v2.x * v3.y) * a2 * a3;
		float d3 = (v3.x * v1.y) * a3 * a1;
		float n1 = (v3.x * v2.y) * a3 * a2;
		float n2 = (v1.x * v3.y) * a1 * a3;
		float n3 = (v2.x * v1.y) * a2 * a1;
		float d = d1 + d2 + d3 - n1 - n2 - n3;
		return d > 0.0f;
	}

	IFRIT_DEVICE void devGetAcceptRejectCoords(ifloat3 edgeCoefs[3], int chosenCoordTR[3], int chosenCoordTA[3]) {
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

	IFRIT_DEVICE bool devTriangleSimpleClip(const ifloat4& v1, const ifloat4& v2, const ifloat4& v3, irect2Df& bbox) {
		bool inside = true;
		float minx = min(v1.x, min(v2.x, v3.x));
		float miny = min(v1.y, min(v2.y, v3.y));
		float maxx = max(v1.x, max(v2.x, v3.x));
		float maxy = max(v1.y, max(v2.y, v3.y));
		float maxz = max(v1.z, max(v2.z, v3.z));
		float minz = min(v1.z, min(v2.z, v3.z));
		if (maxz < 0.0f) return false;
		if (minz > 1.0f) return false;
		if (maxx < 0.0f) return false;
		if (minx > 1.0f) return false;
		if (maxy < 0.0f) return false;
		if (miny > 1.0f) return false;
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
		for (int y = tileLargeMiny; y <= tileLargeMaxy; y++) {

			auto curTileLargeY = y * frameBufferHeight / CU_LARGE_BIN_SIZE;
			auto curTileLargeY2 = (y + 1) * frameBufferHeight / CU_LARGE_BIN_SIZE;
			auto cty1 = 1.0f * curTileLargeY, cty2 = 1.0f * (curTileLargeY2 - 1);

			for (int x = tileLargeMinx; x <= tileLargeMaxx; x++) {

				auto curTileLargeX = x * frameBufferWidth / CU_LARGE_BIN_SIZE;
				auto curTileLargeX2 = (x + 1) * frameBufferWidth / CU_LARGE_BIN_SIZE;
				auto ctx1 = 1.0f * curTileLargeX;
				auto ctx2 = 1.0f * (curTileLargeX2 - 1);

				tileCoordsT[VLT] = { ctx1, cty1 };
				tileCoordsT[VLB] = { ctx1, cty2 };
				tileCoordsT[VRB] = { ctx2, cty2 };
				tileCoordsT[VRT] = { ctx2, cty1 };

				int criteriaTR = 0, criteriaTA = 0;
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoordsT[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoordsT[chosenCoordTR[i]].y;
					float criteriaTALocal = edgeCoefs[i].x * tileCoordsT[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoordsT[chosenCoordTA[i]].y;
					criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
					criteriaTA += criteriaTALocal < edgeCoefs[i].z;
				}
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

							tileCoordsS[VLT] = { ctx1, cty1 };
							tileCoordsS[VLB] = { ctx1, cty2 };
							tileCoordsS[VRB] = { ctx2, cty2 };
							tileCoordsS[VRT] = { ctx2, cty1 };

							int criteriaTR = 0, criteriaTA = 0;
							for (int i = 0; i < 3; i++) {
								float criteriaTRLocal = edgeCoefs[i].x * tileCoordsS[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoordsS[chosenCoordTR[i]].y;
								float criteriaTALocal = edgeCoefs[i].x * tileCoordsS[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoordsS[chosenCoordTA[i]].y;
								criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
								criteriaTA += criteriaTALocal < edgeCoefs[i].z;
							}
							if (criteriaTR != 3)
								continue;
							auto tileId = ty * CU_BIN_SIZE + tx;
							if (criteriaTA == 3) {
								dCoverQueueFullM2[tileId].push_back(primitiveId);
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

	IFRIT_DEVICE void devExecuteBinner(int primitiveId, const AssembledTriangleProposalCUDA& atp,irect2Df bbox) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;
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

		ifloat2 tileCoords[4];


		int chosenCoordTR[3];
		int chosenCoordTA[3];
		auto frameBufferWidth = csFrameWidth;
		auto frameBufferHeight = csFrameHeight;
		devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

		for (int y = tileMiny; y <= tileMaxy; y++) {

			auto curTileY = y * frameBufferHeight / CU_BIN_SIZE;
			auto curTileY2 = (y + 1) * frameBufferHeight / CU_BIN_SIZE;
			auto cty1 = 1.0f * curTileY,cty2 = 1.0f * (curTileY2 - 1);

			for (int x = tileMinx; x <= tileMaxx; x++) {
				auto curTileX = x * frameBufferWidth / CU_BIN_SIZE;
				auto curTileX2 = (x + 1) * frameBufferWidth / CU_BIN_SIZE;
				auto ctx1 = 1.0f * curTileX;
				auto ctx2 = 1.0f * (curTileX2-1);

				tileCoords[VLT] = { ctx1, cty1 };
				tileCoords[VLB] = { ctx1, cty2 };
				tileCoords[VRB] = { ctx2, cty2 };
				tileCoords[VRT] = { ctx2, cty1 };

				int criteriaTR = 0,criteriaTA = 0;
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoords[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTR[i]].y;
					float criteriaTALocal = edgeCoefs[i].x * tileCoords[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTA[i]].y;
					criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
					criteriaTA += criteriaTALocal < edgeCoefs[i].z;
				}
				if (criteriaTR != 3) 
					continue;
				auto tileId = y * CU_BIN_SIZE + x;
				if (criteriaTA == 3) {
					dCoverQueueFullM2[tileId].push_back(primitiveId);
				}	
				else {
					dRasterQueueM2[tileId].push_back(primitiveId);
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

	IFRIT_DEVICE void devPixelProcessingShadingPass(
		const AssembledTriangleProposalCUDA& atp,
		FragmentShader* fragmentShader,
		float bary[3],
		ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
		const int* IFRIT_RESTRICT_CUDA dIndexBuffer,
		const VaryingStore* const* IFRIT_RESTRICT_CUDA dVaryingBuffer,
		int vertexStride,
		int varyingCount,
		int pixelPos
	) {
		ifloat4 colorOutputSingle;
		VaryingStore interpolatedVaryings[CU_MAX_VARYINGS];
		float desiredBary[3];
		desiredBary[0] = bary[0] * atp.b1.x + bary[1] * atp.b2.x + bary[2] * atp.b3.x;
		desiredBary[1] = bary[0] * atp.b1.y + bary[1] * atp.b2.y + bary[2] * atp.b3.y;
		desiredBary[2] = bary[0] * atp.b1.z + bary[1] * atp.b2.z + bary[2] * atp.b3.z;
		auto addr = dIndexBuffer + atp.originalPrimitive * vertexStride;
		for (int k = 0; k < varyingCount; k++) {
			devInterpolateVaryings(k, dVaryingBuffer, addr, desiredBary, interpolatedVaryings[k]);
		}
		fragmentShader->execute(interpolatedVaryings, &colorOutputSingle);
		dColorBuffer[0][pixelPos] = colorOutputSingle;
	}


	IFRIT_DEVICE void devTilingRasterizationChildProcess(
		uint32_t tileIdX,
		uint32_t tileIdY,
		uint32_t invoId,
		uint32_t totalBound
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		auto globalInvocation = invoId;
		if (globalInvocation > totalBound)return;

		const auto tileId = tileIdY * CU_TILE_SIZE + tileIdX;
		const auto binId = tileIdY / CU_TILES_PER_BIN * CU_BIN_SIZE + tileIdX / CU_TILES_PER_BIN;

		const auto frameWidth = csFrameWidth;
		const auto frameHeight =csFrameHeight;

		const auto primitiveSrcId = dRasterQueueM2[binId].at(globalInvocation);
		const auto& atri = dAssembledTriangleM2[primitiveSrcId];

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = atri.e1;
		edgeCoefs[1] = atri.e2;
		edgeCoefs[2] = atri.e3;

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

		auto curTileX = tileIdX * frameWidth / CU_TILE_SIZE;
		auto curTileY = tileIdY * frameHeight / CU_TILE_SIZE;
		auto curTileX2 = (tileIdX + 1) * frameWidth / CU_TILE_SIZE;
		auto curTileY2 = (tileIdY + 1) * frameHeight / CU_TILE_SIZE;
		auto curTileWid = curTileX2 - curTileX;
		auto curTileHei = curTileY2 - curTileY;

		const float dEps = CU_EPS * frameHeight * frameWidth;

		// Decomp into Sub Blocks
		
		for (int i = CU_SUBTILE_SIZE * CU_SUBTILE_SIZE - 1 - threadIdx.y; i >= 0; i--) {
			int criteriaTR = 0;
			int criteriaTA = 0;

			auto subTileIX = i % CU_SUBTILE_SIZE;
			auto subTileIY = i / CU_SUBTILE_SIZE;
			auto subTileTX = (tileIdX * CU_SUBTILE_SIZE + subTileIX);
			auto subTileTY = (tileIdY * CU_SUBTILE_SIZE + subTileIY);

			const int wp = (CU_SUBTILE_SIZE * CU_TILE_SIZE);
			int subTilePixelX = curTileX + (curTileWid * subTileIX >> CU_SUBTILE_SIZE_LOG);
			int subTilePixelY = curTileY + (curTileHei * subTileIY >> CU_SUBTILE_SIZE_LOG);
			int subTilePixelX2 = curTileX + (curTileWid * (subTileIX + 1) >> CU_SUBTILE_SIZE_LOG);
			int subTilePixelY2 = curTileY + (curTileHei * (subTileIY + 1) >> CU_SUBTILE_SIZE_LOG);

			float subTileMinX = 1.0f * subTilePixelX;
			float subTileMinY = 1.0f * subTilePixelY;
			float subTileMaxX = 1.0f * (subTilePixelX2 - 1);
			float subTileMaxY = 1.0f * (subTilePixelY2 - 1);


			ifloat2 tileCoords[4];
			tileCoords[VLT] = { subTileMinX, subTileMinY };
			tileCoords[VLB] = { subTileMinX, subTileMaxY };
			tileCoords[VRB] = { subTileMaxX, subTileMaxY };
			tileCoords[VRT] = { subTileMaxX, subTileMinY };

			for (int k = 0; k < 3; k++) {
				float criteriaTRLocal = edgeCoefs[k].x * tileCoords[chosenCoordTR[k]].x + edgeCoefs[k].y * tileCoords[chosenCoordTR[k]].y;
				float criteriaTALocal = edgeCoefs[k].x * tileCoords[chosenCoordTA[k]].x + edgeCoefs[k].y * tileCoords[chosenCoordTA[k]].y;
				criteriaTR += criteriaTRLocal < edgeCoefs[k].z;
				criteriaTA += criteriaTALocal < edgeCoefs[k].z;
			}

			if (criteriaTR != 3) {
				continue;
			}
			if (criteriaTA == 3) {
				TileBinProposalCUDA nprop;
				nprop.tileEnd = { (short)(subTilePixelX2 - 1),(short)(subTilePixelY2 - 1) };
				nprop.tile = { (short)subTilePixelX,(short)subTilePixelY };
				nprop.primId = primitiveSrcId;
				dCoverQueuePartialM2[tileId].push_back(nprop);
			}
			else {
				//Into Pixel level
				int wid = subTilePixelX2 - subTilePixelX;
				int hei = subTilePixelY2 - subTilePixelY;
				int tot = wid * hei;
				IFRIT_ASSUME(tot > 0);
				for (int i2 = tot - 1; i2 >= 0; i2--) {
					int dx = subTilePixelX + (uint32_t)i2 % (uint32_t)wid;
					int dy = subTilePixelY + (uint32_t)i2 / (uint32_t)wid;
					int accept = 0;
					for (int i = 0; i < 3; i++) {
						float criteria = edgeCoefs[i].x * dx + edgeCoefs[i].y * dy;
						accept += criteria < edgeCoefs[i].z;
					}
					if (accept == 3) {
						TileBinProposalCUDA nprop;
						nprop.tileEnd = { (short)dx,(short)dy };
						nprop.tile = { (short)dx,(short)dy };
						nprop.primId = primitiveSrcId;
						dCoverQueuePartialM2[tileId].push_back(nprop);
					}
				}
			}
		}
	}
 
	// Kernel Implementations

	IFRIT_KERNEL void vertexProcessingKernel(
		VertexShader* vertexShader,
		uint32_t vertexCount,
		char* dVertexBuffer,
		TypeDescriptorEnum* dVertexTypeDescriptor,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		ifloat4* dPosBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (globalInvoIdx >= vertexCount) return;
		const auto numAttrs = deviceConstants->attributeCount;
		const auto numVaryings = deviceConstants->varyingCount;

		const void* vertexInputPtrs[CU_MAX_ATTRIBUTES];
		VaryingStore* varyingOutputPtrs[CU_MAX_VARYINGS];
		
		for (int i = 0; i < numAttrs; i++) {
			vertexInputPtrs[i] = globalInvoIdx * csTotalVertexOffsets + dVertexBuffer + csVertexOffsets[i];
		}
		for (int i = 0; i < numVaryings; i++) {
			varyingOutputPtrs[i] = dVaryingBuffer[i] + globalInvoIdx;
		}
		auto s = *reinterpret_cast<const ifloat4*>(vertexInputPtrs[0]);
		if (blockIdx.x * blockDim.x + threadIdx.x >= 25015 && blockIdx.x * blockDim.x + threadIdx.x <= 25017) {
			printf("%d In:%f,%f,%f,%f\n", blockIdx.x * blockDim.x + threadIdx.x , s.x, s.y, s.z, s.w);
		}
		vertexShader->execute(vertexInputPtrs, &dPosBuffer[globalInvoIdx], varyingOutputPtrs);

		s = dPosBuffer[globalInvoIdx];
		if (blockIdx.x * blockDim.x + threadIdx.x >= 25015 && blockIdx.x * blockDim.x + threadIdx.x <= 25017) {
			printf("%d PsO:%f,%f,%f,%f\n", blockIdx.x * blockDim.x + threadIdx.x, s.x, s.y, s.z, s.w);
		}
		s.x /= s.w;
		s.y /= s.w;
		s.z /= s.w;
		if (blockIdx.x * blockDim.x + threadIdx.x >= 25015 && blockIdx.x * blockDim.x + threadIdx.x <= 25017) {
			printf("%d PsR:%f,%f,%f,%f\n", blockIdx.x * blockDim.x + threadIdx.x, s.x, s.y, s.z, s.w);
			//dPosBuffer[globalInvoIdx].z = dPosBuffer[globalInvoIdx].w * 0.999925;
		}

	}

	IFRIT_KERNEL void geometryProcessingKernel(
		ifloat4* IFRIT_RESTRICT_CUDA dPosBuffer,
		int* IFRIT_RESTRICT_CUDA dIndexBuffer,
		uint32_t startingIndexId,
		uint32_t indexCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if(globalInvoIdx >= indexCount / CU_TRIANGLE_STRIDE) return;

		const auto indexStart = globalInvoIdx * CU_TRIANGLE_STRIDE + startingIndexId;
		ifloat4 v1 = dPosBuffer[dIndexBuffer[indexStart]];
		ifloat4 v2 = dPosBuffer[dIndexBuffer[indexStart + 1]];
		ifloat4 v3 = dPosBuffer[dIndexBuffer[indexStart + 2]];
		//printf("%d %d %d\n", dIndexBuffer[indexStart], dIndexBuffer[indexStart + 1], dIndexBuffer[indexStart + 2]);
		if (csCounterClosewiseCull) {
			ifloat4 temp = v1;
			v1 = v3;
			v3 = temp;
		}
		const auto primId = globalInvoIdx + startingIndexId / CU_TRIANGLE_STRIDE;
		if (!devTriangleCull(v1, v2, v3)) {
			return;
		}

		using Ifrit::Engine::Math::ShaderOps::CUDA::dot;
		using Ifrit::Engine::Math::ShaderOps::CUDA::sub;
		using Ifrit::Engine::Math::ShaderOps::CUDA::add;
		using Ifrit::Engine::Math::ShaderOps::CUDA::multiply;
		using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;

		constexpr uint32_t clipIts = (CU_OPT_HOMOGENEOUS_CLIPPING_NEG_W_ONLY) ? 1 : 7;
		constexpr int possibleTris = CU_OPT_HOMOGENEOUS_CLIPPING_NEG_W_ONLY ? 4 : 7;

		const ifloat4 clipCriteria[7] = {
			{0,0,0,1e-3},
			{1,0,0,0},
			{-1,0,0,0},
			{0,1,0,0},
			{0,-1,0,0},
			{0,0,1,0},
			{0,0,-1,0}
		};

		TileRasterClipVertexCUDA retd[possibleTris+1];
		int retdIndex[2][possibleTris];
		int retdTriCnt = 3;
		
#define ret(x,y) retd[retdIndex[x][y]]
		
		uint32_t retCnt[2] = { 0,3 };
		retd[0] = { {1,0,0},v1 };
		retd[1] = { {0,1,0},v2 };
		retd[2] = { {0,0,1},v3 };
		retdIndex[1][2] = 2;
		retdIndex[1][0] = 0;
		retdIndex[1][1] = 1;
		int clipTimes = 0;
		for (int i = 0; i < clipIts; i++) {
			if (retdIndex[1][0] > 0x7ff) {
				printf("ERROR1\n");
			}
			ifloat4 outNormal = { clipCriteria[i].x,clipCriteria[i].y,clipCriteria[i].z,-1 };
			ifloat4 refPoint = { clipCriteria[i].x,clipCriteria[i].y,clipCriteria[i].z,clipCriteria[i].w };
			const auto cIdx = i & 1, cRIdx = 1 - (i & 1);
			retCnt[cIdx] = 0;
			const auto psize = retCnt[cRIdx];
			auto pc = ret(cRIdx, 0);
			auto npc = dot(pc.pos, outNormal);
			for (int j = 0; j < psize; j++) {
				const auto pn = ret(cRIdx, (j + 1) % psize);
				auto npn = dot(pn.pos, outNormal);
				if constexpr (!CU_OPT_HOMOGENEOUS_DISCARD) {
					if (npc * npn < 0) {
						ifloat4 dir = sub(pn.pos, pc.pos);
						float numo = pc.pos.w - pc.pos.x * refPoint.x - pc.pos.y * refPoint.y - pc.pos.z * refPoint.z;
						float deno = dir.x * refPoint.x + dir.y * refPoint.y + dir.z * refPoint.z - dir.w;
						float t = (-refPoint.w + numo) / deno;
						ifloat4 intersection = add(pc.pos, multiply(dir, t));
						ifloat3 barycenter = lerp(pc.barycenter, pn.barycenter, t);

						TileRasterClipVertexCUDA newp;
						newp.barycenter = barycenter;
						newp.pos = intersection;

						retd[retdTriCnt] = newp;
						retdIndex[cIdx][retCnt[cIdx]++] = retdTriCnt;
						retdTriCnt++;
					}
				}
				
				if (npn < CU_EPS) {
					retdIndex[cIdx][retCnt[cIdx]++] = (j + 1) % psize;
				}
				npc = npn;
				pc = pn;
			}
			if (retCnt[cIdx] < 3) {
				return;
			}
		}


		const auto clipOdd = clipTimes & 1;
		for (int i = 0; i < retCnt[clipOdd]; i++) {
			
			ret(clipOdd, i).pos.x /= ret(clipOdd, i).pos.w;
			ret(clipOdd, i).pos.y /= ret(clipOdd, i).pos.w;
			ret(clipOdd, i).pos.z /= ret(clipOdd, i).pos.w;

			ret(clipOdd, i).pos.w = 1 / ret(clipOdd, i).pos.w;
			ret(clipOdd, i).pos.x = ret(clipOdd, i).pos.x * 0.5f + 0.5f;
			ret(clipOdd, i).pos.y = ret(clipOdd, i).pos.y * 0.5f + 0.5f;
		}
		// Atomic Insertions
		auto threadId = threadIdx.x;

		const auto frameHeight = csFrameHeight;
		const auto frameWidth = csFrameWidth;

		unsigned idxSrc;
		if constexpr (CU_OPT_PREALLOCATED_TRIANGLE_LIST) {
			idxSrc = globalInvoIdx * 2;
			if constexpr (!CU_OPT_HOMOGENEOUS_CLIPPING_NEG_W_ONLY) {
				printf("Memory limit exceed!\n");
				asm("trap;");
			}
		}
		else {
			idxSrc = atomicAdd(&dAssembledTriangleCounterM2, retCnt[clipOdd] - 2);
		}
		
		const auto invFrameHeight = csFrameHeightInv;
		const auto invFrameWidth = csFrameWidthInv;
		for (int i = 0; i < retCnt[clipOdd] - 2; i++) {
			if (i >= 2) {
				printf("ERROR CLIPPING\n");
				return;
			}
			uint32_t curIdx = idxSrc + i;

			AssembledTriangleProposalCUDA atri;
			atri.b1 = ret(clipOdd, 0).barycenter;
			atri.b2 = ret(clipOdd, i + 1).barycenter;
			atri.b3 = ret(clipOdd, i + 2).barycenter;
			const auto dv2 = ret(clipOdd, i + 1).pos;
			const auto dv3 = ret(clipOdd, i + 2).pos;
			const auto dv1 = ret(clipOdd, 0).pos;
			atri.v1 = dv1.z;
			atri.v2 = dv2.z;
			atri.v3 = dv3.z;
			atri.w1 = dv1.w;
			atri.w2 = dv2.w;
			atri.w3 = dv3.w;

			const float ar = 1.0f / devEdgeFunction(dv1, dv2, dv3);
			const float sV2V1y = dv2.y - dv1.y;
			const float sV2V1x = dv1.x - dv2.x;
			const float sV3V2y = dv3.y - dv2.y;
			const float sV3V2x = dv2.x - dv3.x;
			const float sV1V3y = dv1.y - dv3.y;
			const float sV1V3x = dv3.x - dv1.x;

			atri.f3 = { (float)(sV2V1y * ar), (float)(sV2V1x * ar),(float)((-dv1.x * sV2V1y - dv1.y * sV2V1x) * ar) };
			atri.f1 = { (float)(sV3V2y * ar), (float)(sV3V2x * ar),(float)((-dv2.x * sV3V2y - dv2.y * sV3V2x) * ar) };
			atri.f2 = { (float)(sV1V3y * ar), (float)(sV1V3x * ar),(float)((-dv3.x * sV1V3y - dv3.y * sV1V3x) * ar) };

			const auto dEps = CU_EPS*frameHeight*frameWidth;
			ifloat3 edgeCoefs[3];
			atri.e1 = { (float)(sV2V1y)*frameHeight,  (float)(sV2V1x)*frameWidth ,  (float)(-dv2.x * dv1.y + dv1.x * dv2.y) * frameHeight * frameWidth };
			atri.e2 = { (float)(sV3V2y)*frameHeight,  (float)(sV3V2x)*frameWidth ,  (float)(-dv3.x * dv2.y + dv2.x * dv3.y) * frameHeight * frameWidth };
			atri.e3 = { (float)(sV1V3y)*frameHeight,  (float)(sV1V3x)*frameWidth ,  (float)(-dv1.x * dv3.y + dv3.x * dv1.y) * frameHeight * frameWidth };

			atri.e1.z += dEps;
			atri.e2.z += dEps;
			atri.e3.z += dEps;

			atri.originalPrimitive = primId;
			irect2Df bbox;
			dAssembledTriangleM2[curIdx] = atri;
			if (!devTriangleSimpleClip(dv1, dv2, dv3, bbox));
			float bboxSz = min(bbox.w - bbox.x, bbox.h - bbox.y);
			if (bboxSz > CU_LARGE_TRIANGLE_THRESHOLD) {
				devExecuteBinnerLargeTile(idxSrc + i, atri, bbox);
			}
			else {
				devExecuteBinner(idxSrc + i, atri, bbox);
			}
		}
#undef ret

	}

	IFRIT_KERNEL void secondaryBinnerRasterizerKernel() {
		const auto tileIdxX = blockIdx.x ;
		const auto tileIdxY = blockIdx.y;
		const auto threadX = threadIdx.x;
		const auto blockX = blockDim.x;
		const auto tileId = tileIdxY * CU_TILE_SIZE+ tileIdxX;
		const auto binId = tileIdxY / CU_TILES_PER_BIN * CU_BIN_SIZE + tileIdxX / CU_TILES_PER_BIN;
		const auto sdRastCandidates = dRasterQueueM2[binId].getSize();
		for (int i = threadX; i < sdRastCandidates; i+= blockX) {
			devTilingRasterizationChildProcess(tileIdxX, tileIdxY, i, sdRastCandidates);
		}

	}


	IFRIT_KERNEL void fragmentShadingKernelPerTile(
		FragmentShader*  fragmentShader,
		int* IFRIT_RESTRICT_CUDA dIndexBuffer,
		const VaryingStore* const* IFRIT_RESTRICT_CUDA dVaryingBuffer,
		ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
		float* IFRIT_RESTRICT_CUDA dDepthBuffer,
		const TileRasterDeviceConstants* deviceConstants
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
		const auto candidates = dCoverQueuePartialM2[tileId].getSize();
		const auto completeCandidates =  dCoverQueueFullM2[binId].getSize();
		const auto largeCandidates = dCoverQueueSuperTileFullM2[superTileId].getSize();

		constexpr auto vertexStride = CU_TRIANGLE_STRIDE;
		const auto varyingCount = deviceConstants->varyingCount;

		const int threadX = threadIdx.x;
		const int threadY = threadIdx.y;
		const int blockX = blockDim.x;
		const int blockY = blockDim.y;
		const int bds = blockDim.x * blockDim.y;
		const auto threadId = threadY * bds + threadX;

		const int pixelXS = threadX + tileX * csFrameWidth / CU_TILE_SIZE;
		const int pixelYS = threadY + tileY * csFrameHeight / CU_TILE_SIZE;

		float localDepthBuffer = 1;
		float candidateBary[3];
		int candidatePrim = -1;
		const float compareDepth = dDepthBuffer[pixelYS * frameWidth + pixelXS];
		float pDx = 1.0f * pixelXS / csFrameWidth;
		float pDy = 1.0f * pixelYS / csFrameHeight;

		auto zPrePass = [&](const AssembledTriangleProposalCUDA& atp,  int primId) {
			float pos[3];
			pos[0] = atp.v1;
			pos[1] = atp.v2;
			pos[2] = atp.v3;

			float bary[3];
			float interpolatedDepth;

			bary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z);
			bary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z);
			bary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z);
			interpolatedDepth = bary[0] * pos[0] + bary[1] * pos[1] + bary[2] * pos[2];

			if (interpolatedDepth < localDepthBuffer) {
				localDepthBuffer = interpolatedDepth;
				candidatePrim = primId;

				bary[0] *= atp.w1;
				bary[1] *= atp.w2;
				bary[2] *= atp.w3;
				float zCorr = 1.0f / (bary[0] + bary[1] + bary[2]);
				candidateBary[0] = bary[0] * zCorr;
				candidateBary[1] = bary[1] * zCorr;
				candidateBary[2] = bary[2] * zCorr;
			}
		};

		for (int i = largeCandidates - 1; i >= 0; i--) {
			const auto proposal = dCoverQueueSuperTileFullM2[superTileId].at(i);
			const auto atp = dAssembledTriangleM2[proposal];
			zPrePass(atp, proposal);
		}
		for (int i = completeCandidates - 1; i >= 0; i--) {
			const auto proposal = dCoverQueueFullM2[binId].at(i);
			const auto atp = dAssembledTriangleM2[proposal];
			zPrePass(atp, proposal);
		}
		for (int i = candidates - 1; i >= 0; i--) {
			const auto proposal = dCoverQueuePartialM2[tileId].at(i);
			const auto atp = dAssembledTriangleM2[proposal.primId];
			const auto startX = proposal.tile.x;
			const auto startY = proposal.tile.y;
			const auto endX = proposal.tileEnd.x;
			const auto endY = proposal.tileEnd.y;

			if (startX <= pixelXS && pixelXS <= endX && startY <= pixelYS && pixelYS <= endY) {
				zPrePass(atp, proposal.primId);
			}
		}
		if (candidatePrim != -1 && localDepthBuffer< compareDepth) {
			devPixelProcessingShadingPass(dAssembledTriangleM2[candidatePrim], fragmentShader, candidateBary, dColorBuffer, dIndexBuffer,
				dVaryingBuffer, vertexStride, varyingCount, pixelYS * frameWidth + pixelXS);
			dDepthBuffer[pixelYS * frameWidth + pixelXS] = localDepthBuffer;
		}

		if (threadX == 0) {
			dCoverQueuePartialM2[tileId].clear();
			dAssembledTriangleCounterM2 = 0;
		}
	}

	IFRIT_KERNEL void integratedResetKernel() {
		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
		dAssembledTriangleCounterM2 = 0;
		dCoverQueuePartialM2[globalInvocation].clear();
		if (globalInvocation < CU_LARGE_BIN_SIZE * CU_LARGE_BIN_SIZE) {
			dCoverQueueSuperTileFullM2[globalInvocation].clear();
		}
		if (globalInvocation < CU_BIN_SIZE * CU_BIN_SIZE) {
			dCoverQueueFullM2[globalInvocation].clear();
			dRasterQueueM2[globalInvocation].clear();
		}
	}

	IFRIT_KERNEL void integratedInitKernel() {
		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
		dAssembledTriangleCounterM2 = 0;
		dCoverQueuePartialM2[globalInvocation].initialize();
		if (globalInvocation < CU_LARGE_BIN_SIZE * CU_LARGE_BIN_SIZE) {
			dCoverQueueSuperTileFullM2[globalInvocation].initialize();
		}
		if (globalInvocation < CU_BIN_SIZE * CU_BIN_SIZE) {
			dCoverQueueFullM2[globalInvocation].initialize();
			dRasterQueueM2[globalInvocation].initialize();
		}
	}

	IFRIT_KERNEL void resetLargeTileKernel() {
		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
		if (globalInvocation < CU_LARGE_BIN_SIZE * CU_LARGE_BIN_SIZE) {
			dCoverQueueSuperTileFullM2[globalInvocation].clear();
		}
		if (globalInvocation < CU_BIN_SIZE * CU_BIN_SIZE) {
			dCoverQueueFullM2[globalInvocation].clear();
			dRasterQueueM2[globalInvocation].clear();
		}

	}

	IFRIT_KERNEL void imageResetFloat32Kernel(
		float* dBuffer,
		uint32_t channels,
		float value
	) {
		const auto invoX = blockIdx.x * blockDim.x + threadIdx.x;
		const auto invoY = blockIdx.y * blockDim.y + threadIdx.y;
		if (invoX >= csFrameWidth || invoY >= csFrameHeight) {
			return;
		}
		for(int i=0;i<channels;i++) {
			dBuffer[(invoY * csFrameWidth + invoX) * channels + i] = value;
		}
	}

	IFRIT_KERNEL void imageResetInt32Kernel(
		int* dBuffer,
		uint32_t imageX,
		uint32_t imageY,
		uint32_t channels,
		int value
	) {
		const auto invoX = blockIdx.x * blockDim.x + threadIdx.x;
		const auto invoY = blockIdx.y * blockDim.y + threadIdx.y;
		if (invoX >= imageX || invoY >= imageY) {
			return;
		}
		for (int i = 0; i < channels; i++) {
			dBuffer[(invoY * imageX + invoX) * channels + i] = value;
		}
	}


	IFRIT_KERNEL void testingKernel() {
		printf("Hello World\n");
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

	void testingKernelWrapper() {
		Impl::testingKernel CU_KARG2(4,4) ();
		cudaDeviceSynchronize();
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
	int* getShadingLockDeviceAddr(uint32_t bufferSize, int* dOldBuffer) {
		if (dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		int* dBuffer;
		cudaMalloc(&dBuffer, bufferSize * sizeof(int));
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
		TileRasterDeviceConstants* deviceConstants,
		TileRasterDeviceContext* deviceContext,
		bool doubleBuffering,
		ifloat4** dLastColorBuffer,
		float aggressiveRatio
	) IFRIT_AP_NOTHROW {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		cudaMemcpy(deviceContext->dDeviceConstants, deviceConstants, sizeof(TileRasterDeviceConstants), cudaMemcpyHostToDevice);

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

			
			secondPass = 1;
		}
		
		// Compute
		std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
		const int tileSizeX = (Impl::hsFrameWidth / CU_TILE_SIZE) + ((Impl::hsFrameWidth % CU_TILE_SIZE) != 0);
		const int tileSizeY = (Impl::hsFrameHeight / CU_TILE_SIZE) + ((Impl::hsFrameHeight % CU_TILE_SIZE) != 0);

		constexpr int dispatchThreadsX = 8;
		constexpr int dispatchThreadsY = 8;
		int dispatchBlocksX = (Impl::hsFrameWidth / dispatchThreadsX) + ((Impl::hsFrameWidth % dispatchThreadsX) != 0);
		int dispatchBlocksY = (Impl::hsFrameHeight / dispatchThreadsY) + ((Impl::hsFrameHeight % dispatchThreadsY) != 0);

		Impl::imageResetFloat32Kernel CU_KARG4(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY), 0, computeStream)(
			dDepthBuffer, 1, 255.0f
		);
		for (int i = 0; i < dHostColorBufferSize; i++) {
			Impl::imageResetFloat32Kernel CU_KARG4(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY), 0, computeStream)(
				(float*)dHostColorBuffer[i], 4, 0.0f
			);
		}
		int vertexExecutionBlocks = (deviceConstants->vertexCount / CU_VERTEX_PROCESSING_THREADS) + ((deviceConstants->vertexCount % CU_VERTEX_PROCESSING_THREADS) != 0);
		Impl::vertexProcessingKernel CU_KARG4(vertexExecutionBlocks, CU_VERTEX_PROCESSING_THREADS, 0, computeStream)(
			dVertexShader, deviceConstants->vertexCount, dVertexBuffer, dVertexTypeDescriptor,
			deviceContext->dVaryingBuffer, dVaryingTypeDescriptor, dPositionBuffer, deviceContext->dDeviceConstants
		);
		
		constexpr int totalTiles = CU_TILE_SIZE * CU_TILE_SIZE;
		Impl::integratedResetKernel CU_KARG4(CU_TILE_SIZE, CU_TILE_SIZE, 0, computeStream)();
		int dw = 0;
		for (int i = 0; i < deviceConstants->totalIndexCount; i += CU_SINGLE_TIME_TRIANGLE * 3) {
			dw++;
			if (dw != 47207*2-1)continue; // 94413
			auto indexCount = std::min((int)(CU_SINGLE_TIME_TRIANGLE * 3 * aggressiveRatio), deviceConstants->totalIndexCount - i);
			int geometryExecutionBlocks = (indexCount / CU_TRIANGLE_STRIDE / CU_GEOMETRY_PROCESSING_THREADS) + ((indexCount / CU_TRIANGLE_STRIDE % CU_GEOMETRY_PROCESSING_THREADS) != 0);
			Impl::resetLargeTileKernel CU_KARG4(CU_TILE_SIZE, CU_TILE_SIZE, 0, computeStream)();
			Impl::geometryProcessingKernel CU_KARG4(geometryExecutionBlocks, CU_GEOMETRY_PROCESSING_THREADS, 0, computeStream)(
				dPositionBuffer, dIndexBuffer,i, indexCount, deviceContext->dDeviceConstants);
			Impl::secondaryBinnerRasterizerKernel CU_KARG4(dim3(CU_TILE_SIZE, CU_TILE_SIZE, 1), dim3(CU_RASTERIZATION_THREADS_PER_TILE, 1, 1), 0, computeStream)();
			Impl::fragmentShadingKernelPerTile CU_KARG4(dim3(CU_TILE_SIZE, CU_TILE_SIZE, 1), dim3(tileSizeX, tileSizeY, 1), 0, computeStream) (
				dFragmentShader, dIndexBuffer, deviceContext->dVaryingBuffer, dColorBuffer, dDepthBuffer, deviceContext->dDeviceConstants
			);
		}
		if (!doubleBuffering) {
			cudaDeviceSynchronize();
		}

		// Memory Copy
		std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
		if (doubleBuffering) {
			for (int i = 0; i < dHostColorBufferSize; i++) {
				cudaMemcpyAsync(hColorBuffer[i], dLastColorBuffer[i], Impl::hsFrameWidth * Impl::hsFrameHeight * sizeof(ifloat4), cudaMemcpyDeviceToHost, copyStream);
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
		auto memcpyTimes = std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin).count();
		auto computeTimes = std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count();
		auto copybackTimes = std::chrono::duration_cast<std::chrono::microseconds>(end3 - end2).count();

		static long long w = 0;
		static long long wt = 0;
		w += copybackTimes;
		wt += 1;
		printf("AvgTime:%lld\n", w / wt);
		//printf("Memcpy,Compute,Copyback,Counter: %lld,%lld,%lld,%d\n", memcpyTimes, computeTimes, copybackTimes,cntw);
	}
}