#include "engine/tilerastercuda/TileRasterInvocationCuda.cuh"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterConstantsCuda.h"
namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	IFRIT_DEVICE_CONST static int csFrameWidth = 0;
	IFRIT_DEVICE_CONST static int csFrameHeight = 0;
	IFRIT_DEVICE_CONST static bool csCounterClosewiseCull = false;
	IFRIT_DEVICE_CONST static int csVertexOffsets[CU_MAX_ATTRIBUTES];
	IFRIT_DEVICE_CONST static int csTotalVertexOffsets = 0;

	static int hsFrameWidth = 0;
	static int hsFrameHeight = 0;
	static bool hsCounterClosewiseCull = false;
	static int hsVertexOffsets[CU_MAX_ATTRIBUTES];
	static int hsTotalVertexOffsets = 0;

	IFRIT_DEVICE float devEdgeFunction(ifloat4 a, ifloat4 b, ifloat4 c) {
		return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
	}
	IFRIT_DEVICE bool devTriangleCull(ifloat4 v1, ifloat4 v2, ifloat4 v3) {
		float d1 = (v1.x * v2.y);
		float d2 = (v2.x * v3.y);
		float d3 = (v3.x * v1.y);
		float n1 = (v3.x * v2.y);
		float n2 = (v1.x * v3.y);
		float n3 = (v2.x * v1.y);
		float d = d1 + d2 + d3 - n1 - n2 - n3;
		if (d < 0.0f) return false;
		return true;
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

	IFRIT_DEVICE bool devTriangleSimpleClip(ifloat4 v1, ifloat4 v2, ifloat4 v3, irect2Df& bbox) {
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
		bbox.w = maxx - minx;
		bbox.h = maxy - miny;
		return true;
	}
	IFRIT_DEVICE void devExecuteBinner(
		int primitiveId,
		AssembledTriangleProposalCUDA& atp,
		irect2Df bbox,
		uint32_t** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposalCUDA** dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;
		float minx = bbox.x ;
		float miny = bbox.y ;
		float maxx = (bbox.x + bbox.w);
		float maxy = (bbox.y + bbox.h);

		int tileMinx = max(0, (int)(minx * CU_TILE_SIZE));
		int tileMiny = max(0, (int)(miny * CU_TILE_SIZE));
		int tileMaxx = min(CU_TILE_SIZE - 1, (int)(maxx * CU_TILE_SIZE));
		int tileMaxy = min(CU_TILE_SIZE - 1, (int)(maxy * CU_TILE_SIZE));

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = atp.e1;
		edgeCoefs[1] = atp.e2;
		edgeCoefs[2] = atp.e3;

		ifloat2 tileCoords[4];

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		auto frameBufferWidth = csFrameWidth;
		auto frameBufferHeight =csFrameHeight;
		devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

		const float tileSize = 1.0f / CU_TILE_SIZE;
		for (int y = tileMiny; y <= tileMaxy; y++) {

			auto curTileY = y * frameBufferHeight / CU_TILE_SIZE;
			auto curTileY2 = (y + 1) * frameBufferHeight / CU_TILE_SIZE;
			auto cty1 = 1.0f * curTileY;
			auto cty2 = 1.0f * (curTileY2 - 1);

			for (int x = tileMinx; x <= tileMaxx; x++) {
				auto curTileX = x * frameBufferWidth / CU_TILE_SIZE;
				auto curTileX2 = (x + 1) * frameBufferWidth / CU_TILE_SIZE;
				auto ctx1 = 1.0f * curTileX;
				auto ctx2 = 1.0f * (curTileX2-1);

				tileCoords[VLT] = { ctx1, cty1 };
				tileCoords[VLB] = { ctx1, cty2 };
				tileCoords[VRB] = { ctx2, cty2 };
				tileCoords[VRT] = { ctx2, cty1 };

				int criteriaTR = 0;
				int criteriaTA = 0;
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoords[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTR[i]].y;
					float criteriaTALocal = edgeCoefs[i].x * tileCoords[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTA[i]].y;
					if (criteriaTRLocal < -edgeCoefs[i].z) criteriaTR += 1;
					if (criteriaTALocal < -edgeCoefs[i].z) criteriaTA += 1;
				}
				if (criteriaTR != 3) {
					continue;
				}
				auto tileId = y * CU_TILE_SIZE + x;
				auto proposalId = 0;
				if (criteriaTA == 3) {
					proposalId = atomicAdd(&dCoverQueueCount[tileId], 1);
					proposalId = CU_SINGLE_TIME_TRIANGLE - 1 - proposalId;
				}
				else {
					proposalId = atomicAdd(&dRasterQueueCount[tileId], 1);
				}
				dRasterQueue[tileId][proposalId] = primitiveId;
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
		uint32_t totalBound,
		const AssembledTriangleProposalCUDA* IFRIT_RESTRICT_CUDA dAssembledTriangles,
		const uint32_t* IFRIT_RESTRICT_CUDA dRasterQueue,
		TileBinProposalCUDA* IFRIT_RESTRICT_CUDA dCoverQueue,
		uint32_t* IFRIT_RESTRICT_CUDA dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		auto globalInvocation = invoId;
		if (globalInvocation > totalBound)return;

		const auto tileId = tileIdY * CU_TILE_SIZE + tileIdX;
		const auto frameWidth = csFrameWidth;
		const auto frameHeight =csFrameHeight;

		const auto primitiveSrcId = dRasterQueue[globalInvocation];

		const auto& atri = dAssembledTriangles[primitiveSrcId];

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

			const float cmpf[3] = { dEps - edgeCoefs[0].z,dEps - edgeCoefs[1].z,dEps - edgeCoefs[2].z };
			for (int k = 0; k < 3; k++) {
				float criteriaTRLocal = edgeCoefs[k].x * tileCoords[chosenCoordTR[k]].x + edgeCoefs[k].y * tileCoords[chosenCoordTR[k]].y;
				float criteriaTALocal = edgeCoefs[k].x * tileCoords[chosenCoordTA[k]].x + edgeCoefs[k].y * tileCoords[chosenCoordTA[k]].y;
				criteriaTR += criteriaTRLocal < cmpf[k];
				criteriaTA += criteriaTALocal < cmpf[k];
			}

			if (criteriaTR != 3) {
				continue;
			}
			if (criteriaTA == 3) {
				TileBinProposalCUDA nprop;
				nprop.tileEnd = { (short)(subTilePixelX2 - 1),(short)(subTilePixelY2 - 1) };
				nprop.tile = { (short)subTilePixelX,(short)subTilePixelY };
				nprop.primId = primitiveSrcId;
				auto proposalInsIdx = atomicAdd(dCoverQueueCount, 1);
				dCoverQueue[proposalInsIdx] = nprop;
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
						accept += criteria < cmpf[i];
					}
					if (accept == 3) {
						TileBinProposalCUDA nprop;
						nprop.tileEnd = { (short)dx,(short)dy };
						nprop.tile = { (short)dx,(short)dy };
						nprop.primId = primitiveSrcId;
						auto proposalInsIdx = atomicAdd(dCoverQueueCount, 1);
						dCoverQueue[proposalInsIdx] = nprop;
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
		vertexShader->execute(vertexInputPtrs, &dPosBuffer[globalInvoIdx], varyingOutputPtrs);
	}

	IFRIT_KERNEL void primaryBinnerRasterizerKernel(
		irect2Df* IFRIT_RESTRICT_CUDA dTileBounds,
		uint32_t* IFRIT_RESTRICT_CUDA dRasterQueueCount,
		uint32_t* IFRIT_RESTRICT_CUDA dCoverQueueCount,
		uint32_t** IFRIT_RESTRICT_CUDA dRasterQueue,
		AssembledTriangleProposalCUDA* IFRIT_RESTRICT_CUDA dAssembledTriangles,
		uint32_t* IFRIT_RESTRICT_CUDA dAssembledTriangleCount,
		TileRasterDeviceConstants* deviceConstants
	) {

	}

	IFRIT_KERNEL void geometryProcessingKernel(
		ifloat4* IFRIT_RESTRICT_CUDA dPosBuffer,
		int* IFRIT_RESTRICT_CUDA dIndexBuffer,
		AssembledTriangleProposalCUDA* IFRIT_RESTRICT_CUDA dAssembledTriangles,
		uint32_t* IFRIT_RESTRICT_CUDA dAssembledTriangleCount,
		uint32_t** IFRIT_RESTRICT_CUDA dRasterQueue,
		uint32_t* IFRIT_RESTRICT_CUDA dRasterQueueCount,
		TileBinProposalCUDA** IFRIT_RESTRICT_CUDA dCoverQueue,
		uint32_t* IFRIT_RESTRICT_CUDA dCoverQueueCount,
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

		constexpr uint32_t clipIts = 7;
		const ifloat4 clipCriteria[clipIts] = {
			{0,0,0,CU_EPS},
			{1,0,0,0},
			{-1,0,0,0},
			{0,1,0,0},
			{0,-1,0,0},
			{0,0,1,0},
			{0,0,-1,0}
		};

		TileRasterClipVertexCUDA retd[9];
		int retdIndex[14];
		int retdTriCnt = 3;

#define retidx(x,y) retdIndex[(x)*7+(y)]
#define ret(x,y) retd[retdIndex[(x)*7+(y)]]
		uint32_t retCnt[2] = { 0,3 };
		retd[0] = { {1,0,0},v1 };
		retd[1] = { {0,1,0},v2 };
		retd[2] = { {0,0,1},v3 };
		retidx(1,0) = 0;
		retidx(1,1) = 1;
		retidx(1,2) = 2;
		int clipTimes = 0;
		for (int i = 0; i < clipIts; i++) {
			ifloat4 outNormal = { clipCriteria[i].x,clipCriteria[i].y,clipCriteria[i].z,-1 };
			ifloat4 refPoint = { clipCriteria[i].x,clipCriteria[i].y,clipCriteria[i].z,clipCriteria[i].w };
			const auto cIdx = i & 1, cRIdx = 1 - (i & 1);
			retCnt[cIdx] = 0;
			const auto psize = retCnt[cRIdx];
			auto pc = ret(cRIdx, 0);
			auto npc = dot(pc.pos, outNormal);
			for (int j = 0; j < psize; j++) {
				const auto& pn = ret(cRIdx, (j + 1) % psize);
				auto npn = dot(pn.pos, outNormal);

				if (npc * npn < 0) {
					pc = ret(cRIdx, (j + psize- 1) % psize);
					ifloat4 dir = sub(pn.pos, pc.pos);
					float numo = pc.pos.w - pc.pos.x * refPoint.x - pc.pos.y * refPoint.y - pc.pos.z * refPoint.z;
					float deno = dir.x * refPoint.x + dir.y * refPoint.y + dir.z * refPoint.z - dir.w;
					float t = numo / deno;
					ifloat4 intersection = add(pc.pos, multiply(dir, t));
					ifloat3 barycenter = lerp(pc.barycenter, pn.barycenter, t);

					TileRasterClipVertexCUDA newp;
					newp.barycenter = barycenter;
					newp.pos = intersection;
					retd[retdTriCnt++] = newp;
					retidx(cIdx, retCnt[cIdx]++) = retdTriCnt - 1;
				}
				if (npn < CU_EPS) {
					retidx(cIdx, retCnt[cIdx]++) = (j + 1) % psize;
				}
				npc = npn;
			}
			if (retCnt[cIdx] < 3) {
				return;
			}
		}
		const auto clipOdd = clipTimes & 1;
		for (int i = 0; i < retCnt[clipOdd]; i++) {
			ret(clipOdd, i).pos.w = 1 / ret(clipOdd, i).pos.w;
			ret(clipOdd, i).pos.x *= ret(clipOdd, i).pos.w;
			ret(clipOdd, i).pos.y *= ret(clipOdd, i).pos.w;
			ret(clipOdd, i).pos.z *= ret(clipOdd, i).pos.w;

			ret(clipOdd, i).pos.x = ret(clipOdd, i).pos.x * 0.5f + 0.5f;
			ret(clipOdd, i).pos.y = ret(clipOdd, i).pos.y * 0.5f + 0.5f;
		}
		// Atomic Insertions
		auto threadId = threadIdx.x;

		const auto frameHeight = csFrameHeight;
		const auto frameWidth = csFrameWidth;

		auto idxSrc = atomicAdd(dAssembledTriangleCount, retCnt[clipOdd] - 2);
		const auto invFrameHeight = 1.0f / frameHeight;
		const auto invFrameWidth = 1.0f / frameWidth;
		for (int i = 0; i < retCnt[clipOdd] - 2; i++) {
			auto curIdx = idxSrc + i;
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

			const float ar = 1.0f / devEdgeFunction(dv1, dv2, dv3);;
			const float sV2V1y = dv2.y - dv1.y;
			const float sV2V1x = dv1.x - dv2.x;
			const float sV3V2y = dv3.y - dv2.y;
			const float sV3V2x = dv2.x - dv3.x;
			const float sV1V3y = dv1.y - dv3.y;
			const float sV1V3x = dv3.x - dv1.x;

			atri.f3 = { (float)(sV2V1y * ar) * dv3.w * invFrameHeight, (float)(sV2V1x * ar) * dv3.w * invFrameWidth,(float)((-dv1.x * sV2V1y - dv1.y * sV2V1x) * ar) * dv3.w };
			atri.f1 = { (float)(sV3V2y * ar) * dv1.w * invFrameHeight, (float)(sV3V2x * ar) * dv1.w * invFrameWidth,(float)((-dv2.x * sV3V2y - dv2.y * sV3V2x) * ar) * dv1.w };
			atri.f2 = { (float)(sV1V3y * ar) * dv2.w * invFrameHeight, (float)(sV1V3x * ar) * dv2.w * invFrameWidth,(float)((-dv3.x * sV1V3y - dv3.y * sV1V3x) * ar) * dv2.w };

			ifloat3 edgeCoefs[3];
			atri.e1 = { (float)(sV2V1y)*frameHeight,  (float)(sV2V1x)*frameWidth ,  (float)(dv2.x * dv1.y - dv1.x * dv2.y) * frameHeight * frameWidth };
			atri.e2 = { (float)(sV3V2y)*frameHeight,  (float)(sV3V2x)*frameWidth ,  (float)(dv3.x * dv2.y - dv2.x * dv3.y) * frameHeight * frameWidth };
			atri.e3 = { (float)(sV1V3y)*frameHeight,  (float)(sV1V3x)*frameWidth ,  (float)(dv1.x * dv3.y - dv3.x * dv1.y) * frameHeight * frameWidth };

			atri.originalPrimitive = primId;
			irect2Df bbox;
			if (!devTriangleSimpleClip(dv1, dv2, dv3, bbox)) continue;
			if constexpr (CU_NOT_OPT_TILED_BINNER) {
				devExecuteBinner(idxSrc + i, atri, bbox, dRasterQueue, dRasterQueueCount, dCoverQueue, dCoverQueueCount, deviceConstants);
			}
			dAssembledTriangles[curIdx] = atri;
		}
#undef ret
#undef retidx

	}

	IFRIT_KERNEL void secondaryBinnerRasterizerKernel(
		AssembledTriangleProposalCUDA* IFRIT_RESTRICT_CUDA dAssembledTriangles,
		uint32_t* IFRIT_RESTRICT_CUDA dAssembledTriangleCount,
		uint32_t** IFRIT_RESTRICT_CUDA dRasterQueue,
		uint32_t* IFRIT_RESTRICT_CUDA dRasterQueueCount,
		TileBinProposalCUDA** IFRIT_RESTRICT_CUDA dCoverQueue,
		uint32_t* IFRIT_RESTRICT_CUDA dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto tileIdxX = blockIdx.x ;
		const auto tileIdxY = blockIdx.y;
		const auto threadX = threadIdx.x;
		const auto blockX = blockDim.x;
		const auto tileId = tileIdxY * CU_TILE_SIZE+ tileIdxX;
		IFRIT_SHARED uint32_t sdAtomicCounter[1];
		const auto oldCounter = dCoverQueueCount[tileId];
		if (threadX == 0) {
			sdAtomicCounter[0] = 0;
		}
		__syncthreads();
		const auto dRaster = dRasterQueue[tileId];
		const auto dCover = dCoverQueue[tileId];
		const auto sdRastCandidates = dRasterQueueCount[tileId];

		for (int i = threadX; i < sdRastCandidates; i+= blockX) {
			devTilingRasterizationChildProcess(tileIdxX, tileIdxY, i, sdRastCandidates, dAssembledTriangles,
				dRaster, dCover, sdAtomicCounter, deviceConstants);
			
		}
		__syncthreads();
		if (threadX == 0) {
			dCoverQueueCount[tileId] = sdAtomicCounter[0];
			dRasterQueueCount[tileId] = oldCounter;
		}
	}


	IFRIT_KERNEL void fragmentShadingKernelPerTile(
		FragmentShader*  fragmentShader,
		int* IFRIT_RESTRICT_CUDA dIndexBuffer,
		const VaryingStore* const* IFRIT_RESTRICT_CUDA dVaryingBuffer,
		const TileBinProposalCUDA* const* IFRIT_RESTRICT_CUDA dCoverQueue,
		uint32_t** IFRIT_RESTRICT_CUDA dRasterQueue,
		uint32_t* IFRIT_RESTRICT_CUDA dCoverQueueCount,
		uint32_t* IFRIT_RESTRICT_CUDA dRasterQueueCount,
		uint32_t* IFRIT_RESTRICT_CUDA dAssembleTriangleCounter,
		const AssembledTriangleProposalCUDA* IFRIT_RESTRICT_CUDA dAtp,
		ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
		float* IFRIT_RESTRICT_CUDA dDepthBuffer,
		const TileRasterDeviceConstants* deviceConstants
	) {
		uint32_t tileX = blockIdx.x;
		uint32_t tileY = blockIdx.y;

		uint32_t tileId = tileY * CU_TILE_SIZE + tileX;
		const auto frameWidth = csFrameWidth;
		const auto frameHeight = csFrameHeight;
		const auto candidates = dCoverQueueCount[tileId];
		const auto completeCandidates = dRasterQueueCount[tileId];
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
		
		const TileBinProposalCUDA* sdCoverQueueSrc = dCoverQueue[tileId];
		const uint32_t* sdRasterQueueSrc = dRasterQueue[tileId];

		float localDepthBuffer = 1;
		float candidateBary[3];
		int candidatePrim = -1;
		const float compareDepth = dDepthBuffer[pixelYS * frameWidth + pixelXS];
		float pDx = 1.0f * pixelXS;
		float pDy = 1.0f * pixelYS;

		for (int i = completeCandidates - 1; i >= 0; i--) {
			const auto proposal = sdRasterQueueSrc[CU_SINGLE_TIME_TRIANGLE - 1 - i];
			const auto atp = dAtp[proposal];
			float pos[4];
			pos[0] = atp.v1;
			pos[1] = atp.v2;
			pos[2] = atp.v3;

			float bary[3];
			float interpolatedDepth;

			bary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z);
			bary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z);
			bary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z);
			interpolatedDepth = bary[0] * pos[0] + bary[1] * pos[1] + bary[2] * pos[2];
			float zCorr = 1.0f / (bary[0] + bary[1] + bary[2]);
			interpolatedDepth *= zCorr;
			if (interpolatedDepth <= localDepthBuffer) {
				localDepthBuffer = interpolatedDepth;
				candidatePrim = proposal;
				candidateBary[0] = bary[0] * zCorr;
				candidateBary[1] = bary[1] * zCorr;
				candidateBary[2] = bary[2] * zCorr;
			}
		}
		for (int i = candidates - 1; i >= 0; i--) {
			const auto proposal = sdCoverQueueSrc[i];
			const auto atp = dAtp[proposal.primId];
			const auto startX = proposal.tile.x;
			const auto startY = proposal.tile.y;
			const auto endX = proposal.tileEnd.x;
			const auto endY = proposal.tileEnd.y;

			if (startX <= pixelXS && pixelXS <= endX && startY <= pixelYS && pixelYS <= endY) {
				// Z PrePas
				float pos[4];
				pos[0] = atp.v1;
				pos[1] = atp.v2;
				pos[2] = atp.v3;

				float bary[3];
				float interpolatedDepth;

				bary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z);
				bary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z);
				bary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z);
				interpolatedDepth = bary[0] * pos[0] + bary[1] * pos[1] + bary[2] * pos[2];
				float zCorr = 1.0f / (bary[0] + bary[1] + bary[2]);
				interpolatedDepth *= zCorr;
				if (interpolatedDepth <= localDepthBuffer) {
					localDepthBuffer = interpolatedDepth;
					candidatePrim = proposal.primId;
					candidateBary[0] = bary[0] * zCorr;
					candidateBary[1] = bary[1] * zCorr;
					candidateBary[2] = bary[2] * zCorr;
				}
			}
		}
		if (candidatePrim != -1 && localDepthBuffer< compareDepth) {
			devPixelProcessingShadingPass(dAtp[candidatePrim], fragmentShader, candidateBary, dColorBuffer, dIndexBuffer,
				dVaryingBuffer, vertexStride, varyingCount, pixelYS * frameWidth + pixelXS);
			dDepthBuffer[pixelYS * frameWidth + pixelXS] = localDepthBuffer;
		}

		//Reset kernels
		if (threadX == 0) {
			dCoverQueueCount[tileId] = 0;
			dRasterQueueCount[tileId] = 0;
			dAssembleTriangleCounter[0] = 0;
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

	IFRIT_KERNEL void resetKernel(
		uint32_t* count,
		uint32_t size
	) {
		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
		if (globalInvocation >= size) {
			return;
		}
		count[globalInvocation] = 0;
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
		ifloat4** dLastColorBuffer
	) IFRIT_AP_NOTHROW {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		cudaMemcpy(deviceContext->dDeviceConstants, deviceConstants, sizeof(TileRasterDeviceConstants), cudaMemcpyHostToDevice);

		// Stream Preparation
		static int initFlag = 0;
		static cudaStream_t copyStream, computeStream;
		static cudaEvent_t  copyStart, copyEnd;
		if (initFlag == 0) {
			initFlag = 1;
			cudaStreamCreate(&copyStream);
			cudaStreamCreate(&computeStream);
			cudaEventCreate(&copyStart);
			cudaEventCreate(&copyEnd);
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
		Impl::resetKernel CU_KARG4(1, 1, 0, computeStream)(deviceContext->dAssembledTrianglesCounter2, totalTiles);
		Impl::resetKernel CU_KARG4(CU_TILE_SIZE, CU_TILE_SIZE, 0, computeStream)(deviceContext->dRasterQueueCounter, totalTiles);
		Impl::resetKernel CU_KARG4(CU_TILE_SIZE, CU_TILE_SIZE, 0, computeStream)(deviceContext->dCoverQueueCounter, totalTiles);
		for (int i = 0; i < deviceConstants->totalIndexCount; i += CU_SINGLE_TIME_TRIANGLE * 3) {
			auto indexCount = std::min(CU_SINGLE_TIME_TRIANGLE * 3, deviceConstants->totalIndexCount - i);
			int geometryExecutionBlocks = (indexCount / CU_TRIANGLE_STRIDE / CU_GEOMETRY_PROCESSING_THREADS) + ((indexCount / CU_TRIANGLE_STRIDE % CU_GEOMETRY_PROCESSING_THREADS) != 0);
			Impl::geometryProcessingKernel CU_KARG4(geometryExecutionBlocks, CU_GEOMETRY_PROCESSING_THREADS, 0, computeStream)(
				dPositionBuffer, dIndexBuffer, deviceContext->dAssembledTriangles2, deviceContext->dAssembledTrianglesCounter2,
				deviceContext->dRasterQueue,deviceContext->dRasterQueueCounter, deviceContext->dCoverQueue2, deviceContext->dCoverQueueCounter,i, indexCount,
				deviceContext->dDeviceConstants
			);

			Impl::secondaryBinnerRasterizerKernel CU_KARG4(dim3(CU_TILE_SIZE, CU_TILE_SIZE, 1), dim3(CU_RASTERIZATION_THREADS_PER_TILE, 1, 1), 0, computeStream)(
				deviceContext->dAssembledTriangles2, deviceContext->dAssembledTrianglesCounter2,
				deviceContext->dRasterQueue, deviceContext->dRasterQueueCounter, deviceContext->dCoverQueue2, deviceContext->dCoverQueueCounter, 
				deviceContext->dDeviceConstants
			);

			Impl::fragmentShadingKernelPerTile CU_KARG4(dim3(CU_TILE_SIZE, CU_TILE_SIZE, 1), dim3(tileSizeX, tileSizeY, 1), 0, computeStream) (
				dFragmentShader, dIndexBuffer, deviceContext->dVaryingBuffer,
				deviceContext->dCoverQueue2, deviceContext->dRasterQueue, deviceContext->dCoverQueueCounter, deviceContext->dRasterQueueCounter, 
				deviceContext->dAssembledTrianglesCounter2,
				deviceContext->dAssembledTriangles2, dColorBuffer, dDepthBuffer, deviceContext->dDeviceConstants
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