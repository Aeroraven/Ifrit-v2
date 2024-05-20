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
		AssembledTriangleProposal** dProposals, uint32_t* dProposalCount,int* startingIdx) {
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
				return ;
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
		}
		// Atomic Insertions
		auto threadId = threadIdx.x;

		auto idxSrc = atomicAdd(&dProposalCount[threadId], retCnt[clipOdd] - 2);
		for (int i = 0; i < retCnt[clipOdd] - 2; i++) {
			auto curIdx = idxSrc + i;
			AssembledTriangleProposal& atri = dProposals[threadId][curIdx];
			atri.b1 = ret[clipOdd][0].barycenter;
			atri.b2 = ret[clipOdd][i + 1].barycenter;
			atri.b3 = ret[clipOdd][i + 2].barycenter;
			atri.v1 = ret[clipOdd][0].pos;
			atri.v2 = ret[clipOdd][i + 1].pos;
			atri.v3 = ret[clipOdd][i + 2].pos;

			const float ar = 1 / devEdgeFunction(atri.v1, atri.v2, atri.v3);
			const float sV2V1y = atri.v2.y - atri.v1.y;
			const float sV2V1x = atri.v1.x - atri.v2.x;
			const float sV3V2y = atri.v3.y - atri.v2.y;
			const float sV3V2x = atri.v2.x - atri.v3.x;
			const float sV1V3y = atri.v1.y - atri.v3.y;
			const float sV1V3x = atri.v3.x - atri.v1.x;

			atri.f3 = { sV2V1y * ar, sV2V1x * ar,(-atri.v1.x * sV2V1y - atri.v1.y * sV2V1x) * ar };
			atri.f1 = { sV3V2y * ar, sV3V2x * ar,(-atri.v2.x * sV3V2y - atri.v2.y * sV3V2x) * ar };
			atri.f2 = { sV1V3y * ar, sV1V3x * ar,(-atri.v3.x * sV1V3y - atri.v3.y * sV1V3x) * ar };


			ifloat3 edgeCoefs[3];
			atri.e1 = { sV2V1y,  sV2V1x,  atri.v2.x * atri.v1.y - atri.v1.x * atri.v2.y };
			atri.e2 = { sV3V2y,  sV3V2x,  atri.v3.x * atri.v2.y - atri.v2.x * atri.v3.y };
			atri.e3 = { sV1V3y,  sV1V3x,  atri.v1.x * atri.v3.y - atri.v3.x * atri.v1.y };

			atri.originalPrimitive = primitiveId;
			if (primitiveId != 0) {
				printf("ERRORX: %d\n", primitiveId);
			}
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
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal* dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;
		float minx = bbox.x * 0.5 + 0.5;
		float miny = bbox.y * 0.5 + 0.5;
		float maxx = (bbox.x + bbox.w) * 0.5 + 0.5;
		float maxy = (bbox.y + bbox.h) * 0.5 + 0.5;

		int tileMinx = max(0, (int)(minx * CU_TILE_SIZE));
		int tileMiny = max(0, (int)(miny * CU_TILE_SIZE));
		int tileMaxx = min(CU_TILE_SIZE- 1, (int)(maxx * CU_TILE_SIZE));
		int tileMaxy = min(CU_TILE_SIZE- 1, (int)(maxy * CU_TILE_SIZE));

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = atp.e1;
		edgeCoefs[1] = atp.e2;
		edgeCoefs[2] = atp.e3;

		ifloat3 tileCoords[4];

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

				tileCoords[VLT] = { x * tileSize, y * tileSize, 1.0 };
				tileCoords[VLB] = { x * tileSize, (y + 1) * tileSize, 1.0 };
				tileCoords[VRB] = { (x + 1) * tileSize, (y + 1) * tileSize, 1.0 };
				tileCoords[VRT] = { (x + 1) * tileSize, y * tileSize, 1.0 };

				for (int i = 0; i < 4; i++) {
					tileCoords[i].x = tileCoords[i].x * 2 - 1;
					tileCoords[i].y = tileCoords[i].y * 2 - 1;
				}

				int criteriaTR = 0;
				int criteriaTA = 0;
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoords[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTR[i]].y + edgeCoefs[i].z;
					float criteriaTALocal = edgeCoefs[i].x * tileCoords[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTA[i]].y + edgeCoefs[i].z;
					if (criteriaTRLocal < 0) criteriaTR += 1;
					if (criteriaTALocal < 0) criteriaTA += 1;
				}
				if (criteriaTR != 3) {
					//printf("Discard tile %d,%d\n", x, y);
					continue;
				}
				auto workerId = threadIdx.x;
				auto tileId = y * CU_TILE_SIZE+ x;
				if (criteriaTA == 3) {
					TileBinProposal proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.clippedTriangle = { workerId,primitiveId };
					proposal.tile = { x,y };
					auto proposalId = atomicAdd(dCoverQueueCount, 1);
					dCoverQueue[proposalId] = proposal;
					//printf("Covered tile %d,%d\n", x, y);
				}
				else {
					TileBinProposal proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.clippedTriangle = { workerId,primitiveId };
					auto proposalId = atomicAdd(&dRasterQueueCount[tileId], 1);
					dRasterQueue[tileId][proposalId] = proposal;
					//auto proposalId = atomicAdd(&dCoverQueueCount[tileId], 1);
					//dCoverQueue[tileId][proposalId] = proposal;
					//printf("Pending tile %d,%d\n", x, y);
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
	)  {
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
		const void** vertexInputPtrs = new const void* [numAttrs];
		VaryingStore** varyingOutputPtrs = new VaryingStore * [numVaryings];

		int* offsets = new int[numAttrs];
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
		delete[] vertexInputPtrs;
		delete[] varyingOutputPtrs;
		delete[] offsets;
	}

	IFRIT_KERNEL void geometryProcessingKernel(
		ifloat4* dPosBuffer,
		int* dIndexBuffer,
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal* dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if(globalInvoIdx >= deviceConstants->indexCount / deviceConstants->vertexStride) return;
		const auto indexStart = globalInvoIdx * deviceConstants->vertexStride + deviceConstants->startingIndexId;
		ifloat4 v1 = dPosBuffer[dIndexBuffer[indexStart]];
		ifloat4 v2 = dPosBuffer[dIndexBuffer[indexStart + 1]];
		ifloat4 v3 = dPosBuffer[dIndexBuffer[indexStart + 2]];
		if (deviceConstants->counterClockwise) {
			ifloat4 temp = v1;
			v1 = v3;
			v3 = temp;
		}
		const auto primId = (globalInvoIdx + deviceConstants->startingIndexId) / deviceConstants->vertexStride;
		if (primId < 0) {
			printf("ERROR: %d,%d,%d\n", primId, deviceConstants->startingIndexId, globalInvoIdx);
		}
		if (!devTriangleCull(v1, v2, v3)) {
			return;
		}

		int startingIdx;
		int fw = devTriangleHomogeneousClip(primId, v1, v2, v3, dAssembledTriangles, dAssembledTriangleCount, &startingIdx);
		if (fw <=0) {
			return;
		}
		for(int i = startingIdx;i<startingIdx+fw;i++) {
			auto& atri = dAssembledTriangles[threadIdx.x][i];
			irect2Df bbox;
			if(!devTriangleSimpleClip(atri.v1, atri.v2, atri.v3, bbox)) continue;
			
			devExecuteBinner(i, atri, bbox, dRasterQueue, dRasterQueueCount, dCoverQueue, dCoverQueueCount, deviceConstants);
		}
	}


	IFRIT_KERNEL void tilingRasterizationChildKernel(
		uint32_t tileIdX,
		uint32_t tileIdY,
		uint32_t totalBound,
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal* dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;


		const auto tileId = tileIdY * CU_TILE_SIZE+ tileIdX;
		if (globalInvocation > totalBound)return;
		const uint32_t pixelStX = deviceConstants->frameBufferWidth * tileIdX / CU_TILE_SIZE;
		const uint32_t pixelEdX = deviceConstants->frameBufferWidth * (tileIdX + 1) / CU_TILE_SIZE;
		const uint32_t pixelStY = deviceConstants->frameBufferHeight * tileIdY / CU_TILE_SIZE;
		const uint32_t pixelEdY = deviceConstants->frameBufferHeight * (tileIdY + 1) / CU_TILE_SIZE;
		const auto primitiveSrcThread = dRasterQueue[tileId][globalInvocation].clippedTriangle.workerId;
		const auto primitiveSrcId = dRasterQueue[tileId][globalInvocation].clippedTriangle.primId;
		const auto& atri = dAssembledTriangles[primitiveSrcThread][primitiveSrcId];

		float tileMinX = 1.0f * tileIdX / CU_TILE_SIZE;
		float tileMinY = 1.0f * tileIdY / CU_TILE_SIZE;
		float tileMaxX = 1.0f * (tileIdX + 1) / CU_TILE_SIZE;
		float tileMaxY = 1.0f * (tileIdY + 1) / CU_TILE_SIZE;

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = atri.e1;
		edgeCoefs[1] = atri.e2;
		edgeCoefs[2] = atri.e3;

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

		// Decomp into Sub Blocks
		for (int i = CU_SUBTILE_SIZE * CU_SUBTILE_SIZE - 1; i >= 0; --i) {
			int criteriaTR = 0;
			int criteriaTA = 0;

			auto subTileIX = i % CU_SUBTILE_SIZE;
			auto subTileIY = i / CU_SUBTILE_SIZE;
			auto subTileTX = (tileIdX * CU_SUBTILE_SIZE + subTileIX);
			auto subTileTY = (tileIdY * CU_SUBTILE_SIZE + subTileIY);

			const int wp = (CU_SUBTILE_SIZE * CU_TILE_SIZE);
			int subTilePixelX = subTileTX * deviceConstants->frameBufferWidth / wp;
			int subTilePixelY = subTileTY * deviceConstants->frameBufferHeight / wp;
			int subTilePixelX2 = (subTileTX + 1) * deviceConstants->frameBufferWidth / wp;
			int subTilePixelY2 = (subTileTY + 1) * deviceConstants->frameBufferHeight / wp;

			float subTileMinX = tileMinX + 1.0f * subTileIX / wp;
			float subTileMinY = tileMinY + 1.0f * subTileIY / wp;
			float subTileMaxX = tileMinX + 1.0f * (subTileIX + 1) / wp;
			float subTileMaxY = tileMinY + 1.0f * (subTileIY + 1) / wp;

			ifloat3 tileCoords[4];
			tileCoords[VLT] = { subTileMinX, subTileMinY, 1.0 };
			tileCoords[VLB] = { subTileMinX, subTileMaxY, 1.0 };
			tileCoords[VRB] = { subTileMaxX, subTileMaxY, 1.0 };
			tileCoords[VRT] = { subTileMaxX, subTileMinY, 1.0 };

			for (int k = 0; k < 4; k++) {
				tileCoords[k].x = tileCoords[k].x * 2 - 1;
				tileCoords[k].y = tileCoords[k].y * 2 - 1;
			}

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
				TileBinProposal nprop;
				nprop.level = TileRasterLevel::BLOCK;
				nprop.tile = { (int)subTileTX,(int)subTileTY };
				nprop.clippedTriangle = dRasterQueue[tileId][globalInvocation].clippedTriangle;
				auto proposalInsIdx = atomicAdd(dCoverQueueCount, 1);
				dCoverQueue[proposalInsIdx] = nprop;
			}
			else {
				//Into Pixel level
				for (int dx = subTilePixelX; dx <= subTilePixelX2; dx++) {
					for (int dy = subTilePixelY; dy <= subTilePixelY2; dy++) {
						float ndcX = 2.0f * dx / deviceConstants->frameBufferWidth - 1.0f;
						float ndcY = 2.0f * dy / deviceConstants->frameBufferHeight - 1.0f;
						int accept = 0;
						for (int i = 0; i < 3; i++) {
							float criteria = edgeCoefs[i].x * ndcX + edgeCoefs[i].y * ndcY + edgeCoefs[i].z;
							accept += criteria < CU_EPS;
						}
						if (accept == 3) {
							TileBinProposal nprop;
							nprop.level = TileRasterLevel::PIXEL;
							nprop.tile = { dx,dy };
							nprop.clippedTriangle = dRasterQueue[tileId][globalInvocation].clippedTriangle;
							auto proposalInsIdx = atomicAdd(dCoverQueueCount, 1);
							dCoverQueue[proposalInsIdx] = nprop;
						}
					}
				}
			}
		}
	}


	IFRIT_KERNEL void tilingRasterizationKernel(
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal* dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto tileIdxX = blockIdx.x * blockDim.x + threadIdx.x;
		const auto tileIdxY = blockIdx.y * blockDim.y + threadIdx.y;
		const auto tileId = tileIdxY * CU_TILE_SIZE+ tileIdxX;
		const auto rastCandidates = dRasterQueueCount[tileId];

		const auto dispatchBlocks = (rastCandidates >> 5) + ((rastCandidates & 0x1F) != 0);

		if (dispatchBlocks != 0) {
			tilingRasterizationChildKernel CU_KARG2(dispatchBlocks, 32) (tileIdxX, tileIdxY, rastCandidates,
				dAssembledTriangles, dAssembledTriangleCount, dRasterQueue, dRasterQueueCount, dCoverQueue,
				dCoverQueueCount, deviceConstants);
			__syncthreads();
		}
	}


	IFRIT_KERNEL void fragmentShadingKernel(
		FragmentShader* fragmentShader,
		int* dIndexBuffer,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
		ifloat4** dColorBuffer,
		float* dDepthBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {

		// A thread only processes ONE pixel
		const auto pixelX = blockIdx.x * blockDim.x + threadIdx.x;
		const auto pixelY = blockIdx.y * blockDim.y + threadIdx.y;
		if (pixelX >= deviceConstants->frameBufferWidth || pixelY >= deviceConstants->frameBufferHeight) {
			return;
		}

		const auto tileX = pixelX * CU_TILE_SIZE/ deviceConstants->frameBufferWidth;
		const auto tileY = pixelY * CU_TILE_SIZE/ deviceConstants->frameBufferHeight;
		const auto tileId = tileY * CU_TILE_SIZE+ tileX;

		const auto subTileX = pixelX * CU_TILE_SIZE* CU_SUBTILE_SIZE / deviceConstants->frameBufferWidth;
		const auto subTileY = pixelY * CU_TILE_SIZE* CU_SUBTILE_SIZE / deviceConstants->frameBufferHeight;

		//TODO: Time consuming, looping to find related primitives

		VaryingStore* interpolatedVaryings = new VaryingStore[deviceConstants->varyingCount];
		ifloat4 colorOutputSingle;

		for (int i = dCoverQueueCount[tileId]-1; i >= 0; i--) {
			const auto& tileProposal = dCoverQueue[tileId][i];
			if (tileProposal.level == TileRasterLevel::PIXEL &&
				(tileProposal.tile.x != pixelX || tileProposal.tile.y != pixelY)) {
				continue;
			}
			else if (tileProposal.level == TileRasterLevel::BLOCK &&
				(tileProposal.tile.x != subTileX || tileProposal.tile.y != subTileY)) {
				continue;
			}

			// Test accepted
			const auto workerId = tileProposal.clippedTriangle.workerId;
			const auto primId = tileProposal.clippedTriangle.primId;
			const AssembledTriangleProposal& atp = dAssembledTriangles[workerId][primId];

			ifloat4 pos[4];
			pos[0] = atp.v1;
			pos[1] = atp.v2;
			pos[2] = atp.v3;

			float pDx = 2.0f * pixelX / deviceConstants->frameBufferWidth - 1.0f;
			float pDy = 2.0f * pixelY / deviceConstants->frameBufferHeight - 1.0f;

			float bary[3];
			float depth[3];
			float interpolatedDepth;
			const float w[3] = { 1 / pos[0].w,1 / pos[1].w,1 / pos[2].w };
			bary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z) * w[0];
			bary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z) * w[1];
			bary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z) * w[2];
			interpolatedDepth = bary[0] * pos[0].z + bary[1] * pos[1].z + bary[2] * pos[2].z;
			float zCorr = 1.0 / (bary[0] + bary[1] + bary[2]);
			interpolatedDepth *= zCorr;

			auto depthRef = dDepthBuffer[pixelY * deviceConstants->frameBufferWidth + pixelX];
			if (interpolatedDepth > depthRef) {
				continue;
			}
			bary[0] *= zCorr;
			bary[1] *= zCorr;
			bary[2] *= zCorr;

			float desiredBary[3];
			desiredBary[0] = bary[0] * atp.b1.x + bary[1] * atp.b2.x + bary[2] * atp.b3.x;
			desiredBary[1] = bary[0] * atp.b1.y + bary[1] * atp.b2.y + bary[2] * atp.b3.y;
			desiredBary[2] = bary[0] * atp.b1.z + bary[1] * atp.b2.z + bary[2] * atp.b3.z;
			
			//TODO:Check
			auto addr = atp.originalPrimitive * deviceConstants->vertexStride;
			for (int k = 0; k < deviceConstants->varyingCount; k++) {
				devInterpolateVaryings(
					k, dVaryingBuffer, dVaryingTypeDescriptor, 
					dIndexBuffer + addr, desiredBary, interpolatedVaryings[k]);
			}
			fragmentShader->execute(interpolatedVaryings, &colorOutputSingle);
			dColorBuffer[0][pixelY * deviceConstants->frameBufferWidth + pixelX] = colorOutputSingle;
			dDepthBuffer[pixelY * deviceConstants->frameBufferWidth + pixelX] = interpolatedDepth;
		}
		delete[] interpolatedVaryings;
	}

	IFRIT_DEVICE void devPixelShading(
		uint32_t pixelX,
		uint32_t pixelY,
		int* dShadingLock,
		FragmentShader* fragmentShader,
		int* dIndexBuffer,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		AssembledTriangleProposal* dAtp,
		ifloat4** dColorBuffer,
		float* dDepthBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {
		VaryingStore interpolatedVaryings[CU_MAX_VARYINGS];
		ifloat4 colorOutputSingle;
		const AssembledTriangleProposal& atp = *dAtp;
		ifloat4 pos[4];
		pos[0] = atp.v1;
		pos[1] = atp.v2;
		pos[2] = atp.v3;

		float pDx = 2.0f * pixelX / deviceConstants->frameBufferWidth - 1.0f;
		float pDy = 2.0f * pixelY / deviceConstants->frameBufferHeight - 1.0f;

		float bary[3];
		float depth[3];
		float interpolatedDepth;
		const float w[3] = { 1 / pos[0].w,1 / pos[1].w,1 / pos[2].w };
		bary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z) * w[0];
		bary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z) * w[1];
		bary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z) * w[2];
		interpolatedDepth = bary[0] * pos[0].z + bary[1] * pos[1].z + bary[2] * pos[2].z;
		float zCorr = 1.0 / (bary[0] + bary[1] + bary[2]);
		interpolatedDepth *= zCorr;

		
		
		bary[0] *= zCorr;
		bary[1] *= zCorr;
		bary[2] *= zCorr;

		float desiredBary[3];
		desiredBary[0] = bary[0] * atp.b1.x + bary[1] * atp.b2.x + bary[2] * atp.b3.x;
		desiredBary[1] = bary[0] * atp.b1.y + bary[1] * atp.b2.y + bary[2] * atp.b3.y;
		desiredBary[2] = bary[0] * atp.b1.z + bary[1] * atp.b2.z + bary[2] * atp.b3.z;

		//Acquire Lock
		int t = 0;
		while (true) {
			auto f = atomicCAS((dShadingLock + pixelY * deviceConstants->frameBufferWidth + pixelX), 0, 1);
			if (f == 0) break;
			t += 1;
			if (t > 10000000) {
				printf("Deadlock\n");
				return;
			}
		}
		auto depthRef = dDepthBuffer[pixelY * deviceConstants->frameBufferWidth + pixelX];
		if (interpolatedDepth <= depthRef) {
			auto addr = atp.originalPrimitive * deviceConstants->vertexStride;
			for (int k = 0; k < deviceConstants->varyingCount; k++) {
				devInterpolateVaryings(
					k, dVaryingBuffer, dVaryingTypeDescriptor,
					dIndexBuffer + addr, desiredBary, interpolatedVaryings[k]);
			}
			fragmentShader->execute(interpolatedVaryings, &colorOutputSingle);
			dColorBuffer[0][pixelY * deviceConstants->frameBufferWidth + pixelX] = colorOutputSingle;
			dDepthBuffer[pixelY * deviceConstants->frameBufferWidth + pixelX] = interpolatedDepth;
		}
		//Release Lock
		atomicExch((dShadingLock + pixelY * deviceConstants->frameBufferWidth + pixelX), 0);
	}

	IFRIT_KERNEL void pixelShadingGroupKernel(
		uint32_t pixelXS,
		uint32_t pixelYS,
		uint32_t pixelXE,
		uint32_t pixelYE,
		int* dShadingLock,
		FragmentShader* fragmentShader,
		int* dIndexBuffer,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		AssembledTriangleProposal dAtp,
		ifloat4** dColorBuffer,
		float* dDepthBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {
		uint32_t pixelX = pixelXS + threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t pixelY = pixelYS + threadIdx.y + blockIdx.y * blockDim.y;
		if (pixelX >= pixelXE || pixelY >= pixelYE) return;
		
		VaryingStore interpolatedVaryings[CU_MAX_VARYINGS];
		ifloat4 colorOutputSingle;
		
		const AssembledTriangleProposal& atp = dAtp;
		ifloat4 pos[4];
		pos[0] = atp.v1;
		pos[1] = atp.v2;
		pos[2] = atp.v3;

		float pDx = 2.0f * pixelX / deviceConstants->frameBufferWidth - 1.0f;
		float pDy = 2.0f * pixelY / deviceConstants->frameBufferHeight - 1.0f;

		float bary[3];
		float depth[3];
		float interpolatedDepth;
		const float w[3] = { 1 / pos[0].w,1 / pos[1].w,1 / pos[2].w };
		bary[0] = (atp.f1.x * pDx + atp.f1.y * pDy + atp.f1.z) * w[0];
		bary[1] = (atp.f2.x * pDx + atp.f2.y * pDy + atp.f2.z) * w[1];
		bary[2] = (atp.f3.x * pDx + atp.f3.y * pDy + atp.f3.z) * w[2];
		interpolatedDepth = bary[0] * pos[0].z + bary[1] * pos[1].z + bary[2] * pos[2].z;
		float zCorr = 1.0 / (bary[0] + bary[1] + bary[2]);
		interpolatedDepth *= zCorr;

		int t = 0;
		while (true) {
			auto f = atomicCAS((dShadingLock + pixelY * deviceConstants->frameBufferWidth + pixelX), 0, 1);
			if (f == 0) break;
			if (t > 10000000) {
				printf("Deadlock\n");
				return;
			}
		}
		auto depthRef = dDepthBuffer[pixelY * deviceConstants->frameBufferWidth + pixelX];
		if (interpolatedDepth > depthRef) {
			return;
		}
		bary[0] *= zCorr;
		bary[1] *= zCorr;
		bary[2] *= zCorr;

		float desiredBary[3];
		desiredBary[0] = bary[0] * atp.b1.x + bary[1] * atp.b2.x + bary[2] * atp.b3.x;
		desiredBary[1] = bary[0] * atp.b1.y + bary[1] * atp.b2.y + bary[2] * atp.b3.y;
		desiredBary[2] = bary[0] * atp.b1.z + bary[1] * atp.b2.z + bary[2] * atp.b3.z;
		
		//TODO:Check
		auto addr = atp.originalPrimitive * deviceConstants->vertexStride;

		for (int k = 0; k < deviceConstants->varyingCount; k++) {
			devInterpolateVaryings(
				k, dVaryingBuffer, dVaryingTypeDescriptor,
				dIndexBuffer + addr, desiredBary, interpolatedVaryings[k]);
		}
		
		fragmentShader->execute(interpolatedVaryings, &colorOutputSingle);
		dColorBuffer[0][pixelY * deviceConstants->frameBufferWidth + pixelX] = colorOutputSingle;
		dDepthBuffer[pixelY * deviceConstants->frameBufferWidth + pixelX] = interpolatedDepth;
		atomicExch((dShadingLock + pixelY * deviceConstants->frameBufferWidth + pixelX), 0);
	}

	IFRIT_KERNEL void fragmentShadingKernelPerTile(
		FragmentShader* fragmentShader,
		uint32_t* shadingQueue,
		int* dShadingLock,
		int* dIndexBuffer,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal* dCoverQueue,
		uint32_t* dCoverQueueCount,
		ifloat4** dColorBuffer,
		float* dDepthBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {

		while (true) {
			auto shadingCand = devAllocateFragmentShadingCandidate(shadingQueue, dCoverQueueCount, deviceConstants);
			if(shadingCand == -1) break;
			auto proposal = dCoverQueue[shadingCand];
			auto datp = dAssembledTriangles[proposal.clippedTriangle.workerId][proposal.clippedTriangle.primId];
			
			if (proposal.level == TileRasterLevel::PIXEL) {
				const auto pixelX = proposal.tile.x;
				const auto pixelY = proposal.tile.y;
				devPixelShading(pixelX, pixelY, dShadingLock, fragmentShader, dIndexBuffer, dVaryingBuffer,
					dVaryingTypeDescriptor, &datp, dColorBuffer, dDepthBuffer, deviceConstants);
			}
			
			if (proposal.level == TileRasterLevel::TILE) {
				auto tileIdxX = proposal.tile.x;
				auto tileIdxY = proposal.tile.y;
				auto tileId = tileIdxY * CU_TILE_SIZE+ tileIdxX;
				auto curTileX = tileIdxX * deviceConstants->frameBufferWidth / CU_TILE_SIZE;
				auto curTileY = tileIdxY * deviceConstants->frameBufferHeight / CU_TILE_SIZE;
				auto curTileX2 = (tileIdxX + 1) * deviceConstants->frameBufferWidth / CU_TILE_SIZE;
				auto curTileY2 = (tileIdxY + 1) * deviceConstants->frameBufferHeight / CU_TILE_SIZE;
				curTileX2 = min(curTileX2, deviceConstants->frameBufferWidth);
				curTileY2 = min(curTileY2, deviceConstants->frameBufferHeight);

				constexpr auto dispatchThreadsX = 16;
				constexpr auto dispatchThreadsY = 16;
				auto dispatchBlocksX = (curTileX2 - curTileX) / dispatchThreadsX + ((curTileX2 - curTileX) % dispatchThreadsX != 0);
				auto dispatchBlocksY = (curTileY2 - curTileY) / dispatchThreadsY + ((curTileY2 - curTileY) % dispatchThreadsY != 0);
				pixelShadingGroupKernel CU_KARG2(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY)) (
					curTileX, curTileY, curTileX2, curTileY2, dShadingLock, fragmentShader, dIndexBuffer, dVaryingBuffer,
					dVaryingTypeDescriptor, datp, dColorBuffer, dDepthBuffer, deviceConstants);
			}
			if (proposal.level == TileRasterLevel::BLOCK) {
				auto subTilePixelX = (proposal.tile.x) * deviceConstants->frameBufferWidth / CU_TILE_SIZE/ CU_SUBTILE_SIZE;
				auto subTilePixelY = (proposal.tile.y) * deviceConstants->frameBufferHeight / CU_TILE_SIZE/ CU_SUBTILE_SIZE;
				auto subTilePixelX2 = (proposal.tile.x + 1) * deviceConstants->frameBufferWidth / CU_TILE_SIZE/ CU_SUBTILE_SIZE;
				auto subTilePixelY2 = (proposal.tile.y + 1) * deviceConstants->frameBufferHeight / CU_TILE_SIZE/ CU_SUBTILE_SIZE;

				constexpr auto dispatchThreadsX = 8;
				constexpr auto dispatchThreadsY = 8;
				auto dispatchBlocksX = (subTilePixelX2 - subTilePixelX) / dispatchThreadsX + ((subTilePixelX2 - subTilePixelX) % dispatchThreadsX != 0);
				auto dispatchBlocksY = (subTilePixelY2 - subTilePixelY) / dispatchThreadsY + ((subTilePixelY2 - subTilePixelY) % dispatchThreadsY != 0);
				
				pixelShadingGroupKernel CU_KARG2(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY)) (
					subTilePixelX, subTilePixelY, subTilePixelX2, subTilePixelY2, dShadingLock, fragmentShader, dIndexBuffer, dVaryingBuffer,
					dVaryingTypeDescriptor, datp, dColorBuffer, dDepthBuffer, deviceConstants);
			}
			
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

	int invokeCudaRenderingGetTpSize(TypeDescriptorEnum typeDesc) {
		if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT1) {
			return sizeof(float);
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT2) {
			return sizeof(ifloat2);
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT3) {
			return sizeof(ifloat3);
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT4) {
			return sizeof(ifloat4);
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT1) {
			return sizeof(int);
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT2) {
			return sizeof(iint2);
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT3) {
			return sizeof(iint3);
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT4) {
			return sizeof(iint4);
		}
		else {
			return 0;
		}
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
		TileRasterDeviceContext* deviceContext
	) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		cudaMemcpy(deviceContext->dDeviceConstants, deviceConstants, sizeof(TileRasterDeviceConstants), cudaMemcpyHostToDevice);

		// Compute
		std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();

		int dispatchThreadsX = 8;
		int dispatchThreadsY = 8;
		int dispatchBlocksX = (deviceConstants->frameBufferWidth / dispatchThreadsX) + ((deviceConstants->frameBufferWidth % dispatchThreadsX) != 0);
		int dispatchBlocksY = (deviceConstants->frameBufferHeight / dispatchThreadsY) + ((deviceConstants->frameBufferHeight % dispatchThreadsY) != 0);

		Impl::imageResetFloat32Kernel CU_KARG2(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY))(
			dDepthBuffer, deviceConstants->frameBufferWidth, deviceConstants->frameBufferHeight, 1, 255.0f
		);
		for (int i = 0; i < dHostColorBufferSize; i++) {
			Impl::imageResetFloat32Kernel CU_KARG2(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY))(
				(float*)dHostColorBuffer[i], deviceConstants->frameBufferWidth, deviceConstants->frameBufferHeight, 4, 0.0f
				);
		}
		
		Impl::imageResetInt32Kernel CU_KARG2(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY))(
			dShaderLockBuffer, deviceConstants->frameBufferWidth, deviceConstants->frameBufferHeight, 1, 0
		);

		int vertexExecutionBlocks = (deviceConstants->vertexCount / CU_VERTEX_PROCESSING_THREADS) + ((deviceConstants->vertexCount % CU_VERTEX_PROCESSING_THREADS) != 0);
		Impl::vertexProcessingKernel CU_KARG2(vertexExecutionBlocks, CU_VERTEX_PROCESSING_THREADS)(
			dVertexShader, deviceConstants->vertexCount, dVertexBuffer, dVertexTypeDescriptor,
			deviceContext->dVaryingBuffer, dVaryingTypeDescriptor, dPositionBuffer, deviceContext->dDeviceConstants
			);
		
		constexpr int totalTiles = CU_TILE_SIZE * CU_TILE_SIZE;
		Impl::resetKernel CU_KARG2(1,1)(deviceContext->dShadingQueue, 1);
		Impl::resetKernel CU_KARG2(1, CU_GEOMETRY_PROCESSING_THREADS)(deviceContext->dAssembledTrianglesCounter, totalTiles);

		for (int i = 0; i < deviceConstants->totalIndexCount; i += 1000) {
			Impl::resetKernel CU_KARG2(CU_TILE_SIZE, CU_TILE_SIZE)(deviceContext->dRasterQueueCounter, totalTiles);

			deviceConstants->indexCount = std::min(1000, deviceConstants->totalIndexCount - i);
			deviceConstants->startingIndexId = i;
			cudaMemcpy(deviceContext->dDeviceConstants, deviceConstants, sizeof(TileRasterDeviceConstants), cudaMemcpyHostToDevice);

			Impl::resetKernel CU_KARG2(1, 1)(deviceContext->dCoverQueueCounter, 1);
			Impl::resetKernel CU_KARG2(1, CU_GEOMETRY_PROCESSING_THREADS)(deviceContext->dAssembledTrianglesCounter, CU_GEOMETRY_PROCESSING_THREADS);

			int geometryExecutionBlocks = (deviceConstants->indexCount / deviceConstants->vertexStride / CU_GEOMETRY_PROCESSING_THREADS) + ((deviceConstants->indexCount / deviceConstants->vertexStride % CU_GEOMETRY_PROCESSING_THREADS) != 0);
			Impl::geometryProcessingKernel CU_KARG2(geometryExecutionBlocks, CU_GEOMETRY_PROCESSING_THREADS)(
				dPositionBuffer, dIndexBuffer, deviceContext->dAssembledTriangles, deviceContext->dAssembledTrianglesCounter,
				deviceContext->dRasterQueue,deviceContext->dRasterQueueCounter, deviceContext->dCoverQueue, deviceContext->dCoverQueueCounter, deviceContext->dDeviceConstants
				);

			constexpr int tileExecBlockX = (CU_TILE_SIZE / CU_RASTERIZATION_THREADS_PERDIM) + ((CU_TILE_SIZE % CU_RASTERIZATION_THREADS_PERDIM) != 0);
			constexpr int tileExecBlockY = (CU_TILE_SIZE / CU_RASTERIZATION_THREADS_PERDIM) + ((CU_TILE_SIZE % CU_RASTERIZATION_THREADS_PERDIM) != 0);
			Impl::tilingRasterizationKernel CU_KARG2(dim3(tileExecBlockX, tileExecBlockY, 1), dim3(CU_RASTERIZATION_THREADS_PERDIM, CU_RASTERIZATION_THREADS_PERDIM, 1))(
				deviceContext->dAssembledTriangles, deviceContext->dAssembledTrianglesCounter, deviceContext->dRasterQueue, deviceContext->dRasterQueueCounter, deviceContext->dCoverQueue, deviceContext->dCoverQueueCounter, deviceContext->dDeviceConstants
				);

			
			Impl::fragmentShadingKernelPerTile  CU_KARG2(184, 256)(
				dFragmentShader, deviceContext->dShadingQueue, dShaderLockBuffer, dIndexBuffer, deviceContext->dVaryingBuffer, dVaryingTypeDescriptor,
				deviceContext->dAssembledTriangles, deviceContext->dAssembledTrianglesCounter, deviceContext->dCoverQueue, deviceContext->dCoverQueueCounter, dColorBuffer, dDepthBuffer, deviceContext->dDeviceConstants
			);
		}

		cudaDeviceSynchronize();

		std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();

		// Copy back color buffers
		for (int i = 0; i < dHostColorBufferSize; i++) {
			cudaMemcpy(hColorBuffer[i], dHostColorBuffer[i], deviceConstants->frameBufferWidth * deviceConstants->frameBufferHeight * sizeof(ifloat4), cudaMemcpyDeviceToHost);
		}

		std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();

		// Free Memory

		std::chrono::steady_clock::time_point end4 = std::chrono::steady_clock::now();

		auto memcpyTimes = std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin).count();
		auto computeTimes = std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count();
		auto copybackTimes = std::chrono::duration_cast<std::chrono::microseconds>(end3 - end2).count();
		auto freeTimes = std::chrono::duration_cast<std::chrono::microseconds>(end4 - end3).count();

		printf("Memcpy,Compute,Copyback,Free: %lld,%lld,%lld,%lld\n", memcpyTimes, computeTimes, copybackTimes, freeTimes);
		//std::abort();
	}
}