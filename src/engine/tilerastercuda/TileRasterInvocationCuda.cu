#include "engine/tilerastercuda/TileRasterInvocationCuda.cuh"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterConstantsCuda.h"
namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	struct FragmentShadingPos {
		int tileId;
		int candidateId;
	};
	struct FragmentShadingQueue {
		int curTile = 0;
		int curCandidate = 0;
		int lock = 0;
	};

	IFRIT_DEVICE float devEdgeFunction(ifloat4 a, ifloat4 b, ifloat4 c) {
		return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
	}
	IFRIT_DEVICE double devEdgeFunctionDbl(ifloat4 a, ifloat4 b, ifloat4 c) {
		return (1.0*c.x - a.x) * (1.0 * b.y - a.y) - (1.0 * c.y - a.y) * (1.0 * b.x - a.x);
	}
	IFRIT_DEVICE bool devTriangleCull(ifloat4 v1, ifloat4 v2, ifloat4 v3) {
		float d1 = (v1.x * v2.y);
		float d2 = (v2.x * v3.y);
		float d3 = (v3.x * v1.y);
		float n1 = (v3.x * v2.y);
		float n2 = (v1.x * v3.y);
		float n3 = (v2.x * v1.y);
		float d = d1 + d2 + d3 - n1 - n2 - n3;
		if (d < 0.0) return false;
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

	IFRIT_DEVICE int devTriangleHomogeneousClip(const int primitiveId, ifloat4 v1, ifloat4 v2, ifloat4 v3,
		AssembledTriangleProposal* dProposals, uint32_t* dProposalCount, int* startingIdx) {
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
		TileRasterClipVertex ret[2][9];
		uint32_t retCnt[2] = { 0,3 };
		ret[1][0] = { {1,0,0,0},v1 };
		ret[1][1] = { {0,1,0,0},v2 };
		ret[1][2] = { {0,0,1,0},v3 };
		int clipTimes = 0;
		for (int i = 0; i < clipIts; i++) {
			ifloat4 outNormal = { clipCriteria[i].x,clipCriteria[i].y,clipCriteria[i].z,-1 };
			ifloat4 refPoint = { clipCriteria[i].x,clipCriteria[i].y,clipCriteria[i].z,clipCriteria[i].w };
			const auto cIdx = i & 1, cRIdx = 1 - (i & 1);
			retCnt[cIdx] = 0;
			const auto psize = retCnt[cRIdx];
			if (psize == 0) {
				*startingIdx = -1;
				return;
			}
			auto pc = ret[cRIdx][0];
			auto npc = dot(pc.pos, outNormal);
			for (int j = 0; j < psize; j++) {
				const auto& pn = ret[cRIdx][(j + 1) % psize];
				auto npn = dot(pn.pos, outNormal);

				if (npc * npn < 0) {
					ifloat4 dir = sub(pn.pos, pc.pos);

					float numo = pc.pos.w - pc.pos.x * refPoint.x - pc.pos.y * refPoint.y - pc.pos.z * refPoint.z;
					float deno = dir.x * refPoint.x + dir.y * refPoint.y + dir.z * refPoint.z - dir.w;
					float t = numo / deno;
					ifloat4 intersection = add(pc.pos, multiply(dir, t));
					ifloat4 barycenter = lerp(pc.barycenter, pn.barycenter, t);

					TileRasterClipVertex newp;
					newp.barycenter = barycenter;
					newp.pos = intersection;
					ret[cIdx][retCnt[cIdx]++] = (newp);
				}
				if (npn < CU_EPS) {
					ret[cIdx][retCnt[cIdx]++] = pn;
				}
				pc = pn;
				npc = npn;
			}
			if (retCnt[cIdx] < 3) {
				*startingIdx = -1;
				return 0;
			}
		}
		const auto clipOdd = clipTimes & 1;
		for (int i = 0; i < retCnt[clipOdd]; i++) {
			ret[clipOdd][i].pos.x /= ret[clipOdd][i].pos.w;
			ret[clipOdd][i].pos.y /= ret[clipOdd][i].pos.w;
			ret[clipOdd][i].pos.z /= ret[clipOdd][i].pos.w;
			ret[clipOdd][i].pos.w = 1 / ret[clipOdd][i].pos.w;

			ret[clipOdd][i].pos.x = ret[clipOdd][i].pos.x * 0.5 + 0.5;
			ret[clipOdd][i].pos.y = ret[clipOdd][i].pos.y * 0.5 + 0.5;
		}
		// Atomic Insertions
		auto threadId = threadIdx.x;

		auto idxSrc = atomicAdd(dProposalCount, retCnt[clipOdd] - 2);
		for (int i = 0; i < retCnt[clipOdd] - 2; i++) {
			auto curIdx = idxSrc + i;
			AssembledTriangleProposal& atri = dProposals[curIdx];
			atri.b1 = ret[clipOdd][0].barycenter;
			atri.b2 = ret[clipOdd][i + 1].barycenter;
			atri.b3 = ret[clipOdd][i + 2].barycenter;
			atri.v1 = ret[clipOdd][0].pos;
			atri.v2 = ret[clipOdd][i + 1].pos;
			atri.v3 = ret[clipOdd][i + 2].pos;

			const double ar = devEdgeFunctionDbl(atri.v1, atri.v2, atri.v3);
			const double sV2V1y = atri.v2.y - atri.v1.y;
			const double sV2V1x = atri.v1.x - atri.v2.x;
			const double sV3V2y = atri.v3.y - atri.v2.y;
			const double sV3V2x = atri.v2.x - atri.v3.x;
			const double sV1V3y = atri.v1.y - atri.v3.y;
			const double sV1V3x = atri.v3.x - atri.v1.x;

			atri.f3 = { (float)(sV2V1y / ar), (float)(sV2V1x / ar),(float)((-atri.v1.x * sV2V1y - atri.v1.y * sV2V1x) / ar) };
			atri.f1 = { (float)(sV3V2y / ar),(float)(sV3V2x / ar),(float)((-atri.v2.x * sV3V2y - atri.v2.y * sV3V2x) / ar) };
			atri.f2 = { (float)(sV1V3y / ar), (float)(sV1V3x / ar),(float)((-atri.v3.x * sV1V3y - atri.v3.y * sV1V3x) / ar) };


			ifloat3 edgeCoefs[3];
			atri.e1 = { (float)(sV2V1y), (float)(sV2V1x),  (float)(atri.v2.x * atri.v1.y - atri.v1.x * atri.v2.y ) };
			atri.e2 = { (float)(sV3V2y),  (float)(sV3V2x),  (float)(atri.v3.x * atri.v2.y - atri.v2.x * atri.v3.y ) };
			atri.e3 = { (float)(sV1V3y),  (float)(sV1V3x),  (float)(atri.v1.x * atri.v3.y - atri.v3.x * atri.v1.y ) };

			atri.originalPrimitive = primitiveId;
		}
		*startingIdx = idxSrc;
		return  retCnt[clipOdd] - 2;
	}

	IFRIT_DEVICE bool devTriangleSimpleClip(ifloat4 v1, ifloat4 v2, ifloat4 v3, irect2Df& bbox) {
		bool inside = true;
		float minx = min(v1.x, min(v2.x, v3.x));
		float miny = min(v1.y, min(v2.y, v3.y));
		float maxx = max(v1.x, max(v2.x, v3.x));
		float maxy = max(v1.y, max(v2.y, v3.y));
		float maxz = max(v1.z, max(v2.z, v3.z));
		float minz = min(v1.z, min(v2.z, v3.z));
		if (maxz < 0.0) return false;
		if (minz > 1.0) return false;
		if (maxx < -1.0) return false;
		if (minx > 1.0) return false;
		if (maxy < -1.0) return false;
		if (miny > 1.0) return false;
		bbox.x = minx;
		bbox.y = miny;
		bbox.w = maxx - minx;
		bbox.h = maxy - miny;
		return true;
	}

	IFRIT_DEVICE void* devGetBufferAddress(char* dBuffer, TypeDescriptorEnum typeDesc, uint32_t element) {
		if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT1) {
			return reinterpret_cast<float*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT2) {
			return reinterpret_cast<ifloat2*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT3) {
			return reinterpret_cast<ifloat3*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT4) {
			return reinterpret_cast<ifloat4*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT1) {
			return reinterpret_cast<int*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT2) {
			return reinterpret_cast<iint2*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT3) {
			return reinterpret_cast<iint3*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT4) {
			return reinterpret_cast<iint4*>(dBuffer) + element;
		}
		else {
			return nullptr;
		}
	}

	IFRIT_DEVICE void devExecuteBinner(
		int primitiveId,
		AssembledTriangleProposal& atp,
		irect2Df bbox,
		TileBinProposalCUDA** dRasterQueue,
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
		auto frameBufferWidth = deviceConstants->frameBufferWidth;
		auto frameBufferHeight = deviceConstants->frameBufferHeight;
		devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

		const float tileSize = 1.0 / CU_TILE_SIZE;
		for (int y = tileMiny; y <= tileMaxy; y++) {
			for (int x = tileMinx; x <= tileMaxx; x++) {

				auto curTileX = x * frameBufferWidth / CU_TILE_SIZE;
				auto curTileY = y * frameBufferHeight / CU_TILE_SIZE;
				auto curTileX2 = (x + 1) * frameBufferWidth / CU_TILE_SIZE;
				auto curTileY2 = (y + 1) * frameBufferHeight / CU_TILE_SIZE;

				auto ctx1 = 1.0f * curTileX / frameBufferWidth;
				auto ctx2 = 1.0f * (curTileX2-1) / frameBufferWidth;
				auto cty1 = 1.0f * curTileY  / frameBufferHeight;
				auto cty2 = 1.0f * (curTileY2-1) / frameBufferHeight;

				tileCoords[VLT] = { ctx1, cty1 };
				tileCoords[VLB] = { ctx1, cty2 };
				tileCoords[VRB] = { ctx2, cty2 };
				tileCoords[VRT] = { ctx2, cty1 };


				int criteriaTR = 0;
				int criteriaTA = 0;
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoords[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTR[i]].y + edgeCoefs[i].z;
					float criteriaTALocal = edgeCoefs[i].x * tileCoords[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTA[i]].y + edgeCoefs[i].z;
					if (criteriaTRLocal < -CU_EPS) criteriaTR += 1;
					if (criteriaTALocal < CU_EPS) criteriaTA += 1;
				}
				if (criteriaTR != 3) {
					//printf("Discard tile %d,%d\n", x, y);
					continue;
				}
				auto workerId = threadIdx.x;
				auto tileId = y * CU_TILE_SIZE + x;
				if (criteriaTA == 3) {
					TileBinProposalCUDA proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.primId = primitiveId;
					proposal.tile = { x,y };
					auto tileId = y * CU_TILE_SIZE + x;
					auto proposalId = atomicAdd(&dCoverQueueCount[tileId], 1);
					dCoverQueue[tileId][proposalId] = proposal;
				}
				else {
					TileBinProposalCUDA proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.primId = primitiveId;
					auto proposalId = atomicAdd(&dRasterQueueCount[tileId], 1);
					dRasterQueue[tileId][proposalId] = proposal;
				}
			}
		}
	}

	IFRIT_DEVICE void devInterpolateVaryings(
		int id,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		const int indices[3],
		const float barycentric[3],
		VaryingStore& dest
	) {
		auto va = dVaryingBuffer[id];
		auto varyingDescriptor = dVaryingTypeDescriptor[id];

		if (varyingDescriptor == TypeDescriptorEnum::IFTP_FLOAT4) {
			dest.vf4 = { 0,0,0,0 };
			for (int j = 0; j < 3; j++) {
				dest.vf4.x += va[indices[j]].vf4.x * barycentric[j];
				dest.vf4.y += va[indices[j]].vf4.y * barycentric[j];
				dest.vf4.z += va[indices[j]].vf4.z * barycentric[j];
				dest.vf4.w += va[indices[j]].vf4.w * barycentric[j];
			}
		}
		else if (varyingDescriptor == TypeDescriptorEnum::IFTP_FLOAT3) {
			dest.vf3 = { 0,0,0 };
			for (int j = 0; j < 3; j++) {
				dest.vf3.x += va[indices[j]].vf3.x * barycentric[j];
				dest.vf3.y += va[indices[j]].vf3.y * barycentric[j];
				dest.vf3.z += va[indices[j]].vf3.z * barycentric[j];
			}

		}
		else if (varyingDescriptor == TypeDescriptorEnum::IFTP_FLOAT2) {
			dest.vf2 = { 0,0 };
			for (int j = 0; j < 3; j++) {
				dest.vf2.x += va[indices[j]].vf2.x * barycentric[j];
				dest.vf2.y += va[indices[j]].vf2.y * barycentric[j];
			}
		}
		else if (varyingDescriptor == TypeDescriptorEnum::IFTP_FLOAT1) {
			dest.vf = 0;
			for (int j = 0; j < 3; j++) {
				dest.vf += va[indices[j]].vf * barycentric[j];
			}
		}
	}

	IFRIT_DEVICE int devAllocateFragmentShadingCandidate(
		uint32_t* queue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		auto r = atomicAdd(queue, 1);
		if (r >= *dCoverQueueCount) {
			*queue = *dCoverQueueCount;
			return -1;
		}
		return r;
	}


	IFRIT_DEVICE void devPixelShadingUnlocked(
		uint32_t pixelX,
		uint32_t pixelY,
		FragmentShader* fragmentShader,
		int* dIndexBuffer,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		AssembledTriangleProposal dAtp,
		ifloat4** dColorBuffer,
		float* dDepthBuffer,
		int frameBufferWidth,
		int frameBufferHeight,
		int vertexStride,
		int varyingCount
	) {
		VaryingStore interpolatedVaryings[CU_MAX_VARYINGS];
		ifloat4 colorOutputSingle;

		const AssembledTriangleProposal& atp = dAtp;
		ifloat4 pos[4];
		pos[0] = atp.v1;
		pos[1] = atp.v2;
		pos[2] = atp.v3;

		float pDx = 1.0f*pixelX / frameBufferWidth;
		float pDy = 1.0f*pixelY / frameBufferHeight;

		float bary[3];
		float depth[3];
		float interpolatedDepth;

		bary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z) * pos[0].w;
		bary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z) * pos[1].w;
		bary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z) * pos[2].w;
		interpolatedDepth = bary[0] * pos[0].z + bary[1] * pos[1].z + bary[2] * pos[2].z;
		float zCorr = 1.0 / (bary[0] + bary[1] + bary[2]);
		interpolatedDepth *= zCorr;

		bary[0] *= zCorr;
		bary[1] *= zCorr;
		bary[2] *= zCorr;

		const auto pixelPos = pixelY * frameBufferWidth + pixelX;
		auto depthRef = dDepthBuffer[pixelPos];
		if (interpolatedDepth <= depthRef) {
			float desiredBary[3];
			desiredBary[0] = bary[0] * atp.b1.x + bary[1] * atp.b2.x + bary[2] * atp.b3.x;
			desiredBary[1] = bary[0] * atp.b1.y + bary[1] * atp.b2.y + bary[2] * atp.b3.y;
			desiredBary[2] = bary[0] * atp.b1.z + bary[1] * atp.b2.z + bary[2] * atp.b3.z;
			auto addr = dIndexBuffer+atp.originalPrimitive * vertexStride;
			for (int k = 0; k < varyingCount; k++) {
				devInterpolateVaryings(k, dVaryingBuffer, dVaryingTypeDescriptor,addr, desiredBary, interpolatedVaryings[k]);
			}
			fragmentShader->execute(interpolatedVaryings, &colorOutputSingle);
			dColorBuffer[0][pixelPos] = colorOutputSingle;
			//dColorBuffer[0][pixelPos].x += colorOutputSingle.x;
			//dColorBuffer[0][pixelPos].y += colorOutputSingle.y;
			//dColorBuffer[0][pixelPos].z += colorOutputSingle.z;
			dDepthBuffer[pixelPos] = interpolatedDepth;
		}
	}

	IFRIT_DEVICE void devTilingRasterizationChildProcess(
		uint32_t tileIdX,
		uint32_t tileIdY,
		uint32_t invoId,
		uint32_t totalBound,
		AssembledTriangleProposal* dAssembledTriangles,
		TileBinProposalCUDA* IFRIT_RESTRICT_CUDA dRasterQueue,
		TileBinProposalCUDA* IFRIT_RESTRICT_CUDA dCoverQueue,
		uint32_t* IFRIT_RESTRICT_CUDA dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		auto globalInvocation = invoId;
		if (globalInvocation > totalBound)return;

		const auto tileId = tileIdY * CU_TILE_SIZE + tileIdX;
		const auto frameWidth = deviceConstants->frameBufferWidth;
		const auto frameHeight = deviceConstants->frameBufferHeight;

		const uint32_t pixelStX = frameWidth * tileIdX / CU_TILE_SIZE;
		const uint32_t pixelEdX = frameWidth * (tileIdX + 1) / CU_TILE_SIZE;
		const uint32_t pixelStY = frameHeight * tileIdY / CU_TILE_SIZE;
		const uint32_t pixelEdY = frameHeight * (tileIdY + 1) / CU_TILE_SIZE;
		const auto primitiveSrcId = dRasterQueue[globalInvocation].primId;

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

		// Decomp into Sub Blocks
		for (int i = CU_SUBTILE_SIZE * CU_SUBTILE_SIZE - 1; i >= 0; --i) {
			int criteriaTR = 0;
			int criteriaTA = 0;

			auto subTileIX = i % CU_SUBTILE_SIZE;
			auto subTileIY = i / CU_SUBTILE_SIZE;
			auto subTileTX = (tileIdX * CU_SUBTILE_SIZE + subTileIX);
			auto subTileTY = (tileIdY * CU_SUBTILE_SIZE + subTileIY);

			const int wp = (CU_SUBTILE_SIZE * CU_TILE_SIZE);
			int subTilePixelX = curTileX + curTileWid * subTileIX / CU_SUBTILE_SIZE;
			int subTilePixelY = curTileY + curTileHei * subTileIY / CU_SUBTILE_SIZE;
			int subTilePixelX2 = curTileX + curTileWid * (subTileIX + 1) / CU_SUBTILE_SIZE;
			int subTilePixelY2 = curTileY + curTileHei * (subTileIY + 1) / CU_SUBTILE_SIZE;

			float subTileMinX = 1.0 * subTilePixelX / frameWidth;
			float subTileMinY = 1.0 * subTilePixelY / frameHeight;
			float subTileMaxX = 1.0 * (subTilePixelX2-1) / frameWidth;
			float subTileMaxY = 1.0 * (subTilePixelY2-1) / frameHeight;

			ifloat2 tileCoords[4];
			tileCoords[VLT] = { subTileMinX, subTileMinY };
			tileCoords[VLB] = { subTileMinX, subTileMaxY };
			tileCoords[VRB] = { subTileMaxX, subTileMaxY };
			tileCoords[VRT] = { subTileMaxX, subTileMinY };


			for (int k = 0; k < 3; k++) {
				float criteriaTRLocal = edgeCoefs[k].x * tileCoords[chosenCoordTR[k]].x + edgeCoefs[k].y * tileCoords[chosenCoordTR[k]].y + edgeCoefs[k].z;
				float criteriaTALocal = edgeCoefs[k].x * tileCoords[chosenCoordTA[k]].x + edgeCoefs[k].y * tileCoords[chosenCoordTA[k]].y + edgeCoefs[k].z;
				if (criteriaTRLocal < -CU_EPS) criteriaTR += 1;
				if (criteriaTALocal < CU_EPS) criteriaTA += 1;
			}

			if (criteriaTR != 3) {
				continue;
			}
			if (criteriaTA == 3) {
				TileBinProposalCUDA nprop;
				nprop.level = TileRasterLevel::BLOCK;
				nprop.tile = { (int)subTileIX,(int)subTileIY };
				nprop.primId = primitiveSrcId;
				auto proposalInsIdx = atomicAdd(dCoverQueueCount, 1);
				dCoverQueue[proposalInsIdx] = nprop;
			}
			else {
				//Into Pixel level
				const float cmpf[3] = { CU_EPS - edgeCoefs[0].z,CU_EPS - edgeCoefs[1].z,CU_EPS - edgeCoefs[2].z };
				for (int dy = subTilePixelY; dy < subTilePixelY2; dy++) {
					float dyf = 1.0 * dy / frameHeight;
					for (int dx = subTilePixelX; dx < subTilePixelX2; dx++) {
						int accept = 0;
						float dxf = 1.0 * dx / frameWidth;
						for (int i = 0; i < 3; i++) {
							float criteria = edgeCoefs[i].x * dxf + edgeCoefs[i].y * dyf;
							accept += criteria < cmpf[i];
						}
						if (accept == 3) {
							TileBinProposalCUDA nprop;
							nprop.level = TileRasterLevel::PIXEL;
							nprop.tile = { dx,dy };
							nprop.primId = primitiveSrcId;
							auto proposalInsIdx = atomicAdd(dCoverQueueCount, 1);
							dCoverQueue[proposalInsIdx] = nprop;
						}
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

		int offsets[CU_MAX_ATTRIBUTES];
		int totalOffset = 0;
		for (int i = 0; i < numAttrs; i++) {
			int cof = 0;
			offsets[i] = totalOffset;
			if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT1) cof  = sizeof(float);
			else if(dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT2) cof = sizeof(ifloat2);
			else if(dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT3) cof = sizeof(ifloat3);
			else if(dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT4)cof = sizeof(ifloat4);
			else if(dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT1) cof = sizeof(int);
			else if(dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT2) cof = sizeof(iint2);
			else if(dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT3) cof = sizeof(iint3);
			else if(dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT4) cof = sizeof(iint4);
			totalOffset += cof;
		}
		for (int i = 0; i < numAttrs; i++) {
			vertexInputPtrs[i] = globalInvoIdx * totalOffset + dVertexBuffer + offsets[i];
		}
		for (int i = 0; i < numVaryings; i++) {
			varyingOutputPtrs[i] = dVaryingBuffer[i] + globalInvoIdx;
		}
		vertexShader->execute(vertexInputPtrs, &dPosBuffer[globalInvoIdx], varyingOutputPtrs);
	}

	IFRIT_KERNEL void geometryProcessingKernel(
		ifloat4* dPosBuffer,
		int* dIndexBuffer,
		AssembledTriangleProposal* IFRIT_RESTRICT_CUDA dAssembledTriangles,
		uint32_t* IFRIT_RESTRICT_CUDA dAssembledTriangleCount,
		TileBinProposalCUDA** IFRIT_RESTRICT_CUDA dRasterQueue,
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
		if (deviceConstants->counterClockwise) {
			ifloat4 temp = v1;
			v1 = v3;
			v3 = temp;
		}
		const auto primId = globalInvoIdx + startingIndexId / CU_TRIANGLE_STRIDE;
		if (!devTriangleCull(v1, v2, v3)) {
			return;
		}

		int startingIdx;

		int fw = devTriangleHomogeneousClip(primId, v1, v2, v3, dAssembledTriangles, dAssembledTriangleCount, &startingIdx);
		if (fw <=0) {
			return;
		}
		
		for(int i = startingIdx;i<startingIdx+fw;i++) {
			auto& atri = dAssembledTriangles[i];
			irect2Df bbox;
			if(!devTriangleSimpleClip(atri.v1, atri.v2, atri.v3, bbox)) continue;
			devExecuteBinner(i, atri, bbox, dRasterQueue, dRasterQueueCount, dCoverQueue, dCoverQueueCount, deviceConstants);
		}
	}


	IFRIT_KERNEL void tilingRasterizationKernel(
		AssembledTriangleProposal* IFRIT_RESTRICT_CUDA dAssembledTriangles,
		uint32_t* IFRIT_RESTRICT_CUDA dAssembledTriangleCount,
		TileBinProposalCUDA** IFRIT_RESTRICT_CUDA dRasterQueue,
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
		IFRIT_SHARED uint32_t sdRastCandidates;
		if (threadX == 0) {
			sdAtomicCounter[0] = dCoverQueueCount[tileId];
			sdRastCandidates = dRasterQueueCount[tileId];
		}
		__syncthreads();

		for (int i = threadX; i < sdRastCandidates; i+= blockX) {
			devTilingRasterizationChildProcess(tileIdxX, tileIdxY, i, sdRastCandidates, dAssembledTriangles,
				dRasterQueue[tileId], dCoverQueue[tileId], sdAtomicCounter, deviceConstants);
		}
		__syncthreads();
		if (threadX == 0) {
			dCoverQueueCount[tileId] = sdAtomicCounter[0];
		}

	}


	IFRIT_KERNEL void fragmentShadingKernelPerTile(
		FragmentShader*  fragmentShader,
		int* IFRIT_RESTRICT_CUDA dIndexBuffer,
		VaryingStore** IFRIT_RESTRICT_CUDA dVaryingBuffer,
		TypeDescriptorEnum* IFRIT_RESTRICT_CUDA dVaryingTypeDescriptor,
		TileBinProposalCUDA** IFRIT_RESTRICT_CUDA dCoverQueue,
		uint32_t* IFRIT_RESTRICT_CUDA dCoverQueueCount,
		uint32_t* IFRIT_RESTRICT_CUDA dRasterQueueCount,
		uint32_t* IFRIT_RESTRICT_CUDA dAssembleTriangleCounter,
		AssembledTriangleProposal* IFRIT_RESTRICT_CUDA dAtp,
		ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
		float* IFRIT_RESTRICT_CUDA dDepthBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {
		uint32_t tileX = blockIdx.x;
		uint32_t tileY = blockIdx.y;

		uint32_t tileId = tileY * CU_TILE_SIZE + tileX;
		const auto frameWidth = deviceConstants->frameBufferWidth;
		const auto frameHeight = deviceConstants->frameBufferHeight;
		const auto candidates = dCoverQueueCount[tileId];
		const auto vertexStride = CU_TRIANGLE_STRIDE;
		const auto varyingCount = deviceConstants->varyingCount;

		const auto curTileX = tileX * frameWidth / CU_TILE_SIZE;
		const auto curTileY = tileY * frameHeight / CU_TILE_SIZE;
		const auto threadX = threadIdx.x;
		const auto threadY = threadIdx.y;
		constexpr auto bds = CU_FRAGMENT_SHADING_THREADS_PER_TILE_X * CU_FRAGMENT_SHADING_THREADS_PER_TILE_Y;
		const auto threadId = threadY * bds + threadX;
		const auto blockX = blockDim.x;
		const auto blockY = blockDim.y;

		
		IFRIT_SHARED TypeDescriptorEnum sdVaryingTypeDescriptor[CU_MAX_VARYINGS];
		if(threadId<varyingCount) {
			sdVaryingTypeDescriptor[threadId] = dVaryingTypeDescriptor[threadId];
		}
		__syncthreads();

		for (int i = 0; i < candidates; i++) {
			auto proposal = dCoverQueue[tileId][i];
			const auto& atp = dAtp[proposal.primId];
			if (proposal.level == TileRasterLevel::PIXEL) {
				const auto pixelX = proposal.tile.x;
				const auto pixelY = proposal.tile.y;
		
				if (pixelX % blockX != threadX || pixelY % blockY != threadY) continue;
				devPixelShadingUnlocked(pixelX, pixelY, fragmentShader, dIndexBuffer, dVaryingBuffer,
					sdVaryingTypeDescriptor, atp, dColorBuffer, dDepthBuffer,
					frameWidth, frameHeight, vertexStride, varyingCount);
			}
			else if (proposal.level == TileRasterLevel::BLOCK) {
				auto curTileX = tileX * frameWidth / CU_TILE_SIZE;
				auto curTileY = tileY * frameHeight / CU_TILE_SIZE;
				auto curTileX2 = (tileX + 1) * frameWidth / CU_TILE_SIZE;
				auto curTileY2 = (tileY + 1) * frameHeight / CU_TILE_SIZE;

				auto curTileWid = curTileX2 - curTileX;
				auto curTileHei = curTileY2 - curTileY;

				auto subTileX = curTileX + curTileWid * (proposal.tile.x) / CU_SUBTILE_SIZE;
				auto subTileY = curTileY + curTileHei * (proposal.tile.y) / CU_SUBTILE_SIZE;
				auto subTileX2 = curTileX + curTileWid * (proposal.tile.x + 1) / CU_SUBTILE_SIZE;
				auto subTileY2 = curTileY + curTileHei * (proposal.tile.y + 1) / CU_SUBTILE_SIZE;
				for (int dx = subTileX; dx < subTileX2; dx++) {
					for (int dy = subTileY; dy < subTileY2; dy++) {

						if (dx % blockX != threadX || dy % blockY != threadY) continue;
						devPixelShadingUnlocked(dx, dy, fragmentShader, dIndexBuffer, dVaryingBuffer,
							sdVaryingTypeDescriptor, atp, dColorBuffer, dDepthBuffer,
							frameWidth,frameHeight, vertexStride,varyingCount);
					}
				}
			}
			else {
				auto tileId = tileY * CU_TILE_SIZE + tileX;
				auto curTileX = tileX * frameWidth / CU_TILE_SIZE;
				auto curTileY = tileY * frameHeight / CU_TILE_SIZE;
				auto curTileX2 = (tileX + 1) * frameWidth / CU_TILE_SIZE;
				auto curTileY2 = (tileY + 1) * frameHeight / CU_TILE_SIZE;
				curTileX2 = min(curTileX2, frameWidth);
				curTileY2 = min(curTileY2, frameHeight);
				for (int dx = curTileX; dx < curTileX2; dx++) {
					for (int dy = curTileY; dy < curTileY2; dy++) {
						if (dx % blockX != threadX || dy % blockY != threadY) continue;
						devPixelShadingUnlocked(dx, dy, fragmentShader, dIndexBuffer, dVaryingBuffer,
							sdVaryingTypeDescriptor, atp, dColorBuffer, dDepthBuffer,
							frameWidth, frameHeight, vertexStride, varyingCount);
					}
				}
			}
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
		uint32_t imageX,
		uint32_t imageY,
		uint32_t channels,
		float value
	) {
		const auto invoX = blockIdx.x * blockDim.x + threadIdx.x;
		const auto invoY = blockIdx.y * blockDim.y + threadIdx.y;
		if (invoX >= imageX || invoY >= imageY) {
			return;
		}
		for(int i=0;i<channels;i++) {
			dBuffer[(invoY * imageX + invoX) * channels + i] = value;
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
	) {
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

		constexpr int dispatchThreadsX = 8;
		constexpr int dispatchThreadsY = 8;
		int dispatchBlocksX = (deviceConstants->frameBufferWidth / dispatchThreadsX) + ((deviceConstants->frameBufferWidth % dispatchThreadsX) != 0);
		int dispatchBlocksY = (deviceConstants->frameBufferHeight / dispatchThreadsY) + ((deviceConstants->frameBufferHeight % dispatchThreadsY) != 0);

		Impl::imageResetFloat32Kernel CU_KARG4(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY), 0, computeStream)(
			dDepthBuffer, deviceConstants->frameBufferWidth, deviceConstants->frameBufferHeight, 1, 255.0f
		);
		for (int i = 0; i < dHostColorBufferSize; i++) {
			Impl::imageResetFloat32Kernel CU_KARG4(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY), 0, computeStream)(
				(float*)dHostColorBuffer[i], deviceConstants->frameBufferWidth, deviceConstants->frameBufferHeight, 4, 0.0f
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
			Impl::tilingRasterizationKernel CU_KARG4(dim3(CU_TILE_SIZE, CU_TILE_SIZE, 1), dim3(CU_RASTERIZATION_THREADS_PER_TILE, 1, 1), 0, computeStream)(
				deviceContext->dAssembledTriangles2, deviceContext->dAssembledTrianglesCounter2,
				deviceContext->dRasterQueue, deviceContext->dRasterQueueCounter, deviceContext->dCoverQueue2, deviceContext->dCoverQueueCounter, deviceContext->dDeviceConstants
				);
			Impl::fragmentShadingKernelPerTile CU_KARG4(dim3(CU_TILE_SIZE, CU_TILE_SIZE, 1), dim3(CU_FRAGMENT_SHADING_THREADS_PER_TILE_X, CU_FRAGMENT_SHADING_THREADS_PER_TILE_Y, 1), 0, computeStream) (
				dFragmentShader, dIndexBuffer, deviceContext->dVaryingBuffer, dVaryingTypeDescriptor,
				deviceContext->dCoverQueue2, deviceContext->dCoverQueueCounter, deviceContext->dRasterQueueCounter,deviceContext->dAssembledTrianglesCounter2,
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
				cudaMemcpyAsync(hColorBuffer[i], dLastColorBuffer[i], deviceConstants->frameBufferWidth * deviceConstants->frameBufferHeight * sizeof(ifloat4), cudaMemcpyDeviceToHost, copyStream);
			}
		}
		cudaStreamSynchronize(computeStream);
		if (doubleBuffering) {
			cudaStreamSynchronize(copyStream);
		}

		if (!doubleBuffering) {
			for (int i = 0; i < dHostColorBufferSize; i++) {
				cudaMemcpy(hColorBuffer[i], dHostColorBuffer[i], deviceConstants->frameBufferWidth * deviceConstants->frameBufferHeight * sizeof(ifloat4), cudaMemcpyDeviceToHost);
			}
		}


		std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();

		// End of rendering

		auto memcpyTimes = std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin).count();
		auto computeTimes = std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count();
		auto copybackTimes = std::chrono::duration_cast<std::chrono::microseconds>(end3 - end2).count();

		printf("Memcpy,Compute,Copyback: %lld,%lld,%lld\n", memcpyTimes, computeTimes, copybackTimes);
	}
}