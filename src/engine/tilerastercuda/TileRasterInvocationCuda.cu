#include "engine/tilerastercuda/TileRasterInvocationCuda.cuh"
#include "engine/math/ShaderOpsCuda.cuh"
namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
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
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;
		float minx = bbox.x * 0.5 + 0.5;
		float miny = bbox.y * 0.5 + 0.5;
		float maxx = (bbox.x + bbox.w) * 0.5 + 0.5;
		float maxy = (bbox.y + bbox.h) * 0.5 + 0.5;

		int tileMinx = max(0, (int)(minx * deviceConstants->tileBlocksX));
		int tileMiny = max(0, (int)(miny * deviceConstants->tileBlocksX));
		int tileMaxx = min(deviceConstants->tileBlocksX - 1, (int)(maxx * deviceConstants->tileBlocksX));
		int tileMaxy = min(deviceConstants->tileBlocksX - 1, (int)(maxy * deviceConstants->tileBlocksX));

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


		const float tileSize = 1.0 / deviceConstants->tileBlocksX;
		for (int y = tileMiny; y <= tileMaxy; y++) {
			for (int x = tileMinx; x <= tileMaxx; x++) {

				auto curTileX = x * frameBufferWidth / deviceConstants->tileBlocksX;
				auto curTileY = y * frameBufferHeight / deviceConstants->tileBlocksX;
				auto curTileX2 = (x + 1) * frameBufferWidth / deviceConstants->tileBlocksX;
				auto curTileY2 = (y + 1) * frameBufferHeight / deviceConstants->tileBlocksX;

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
				if (criteriaTR != 3)continue;
				auto workerId = threadIdx.x;
				auto tileId = y * deviceConstants->tileBlocksX + x;
				if (criteriaTA == 3) {
					TileBinProposal proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.clippedTriangle = { workerId,primitiveId };
					auto proposalId = atomicAdd(&dCoverQueueCount[tileId], 1);
					dCoverQueue[tileId][proposalId] = proposal;
				}
				else {
					TileBinProposal proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.clippedTriangle = { workerId,primitiveId };
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

	// Kernel Implementations

	IFRIT_KERNEL void vertexProcessingKernel(
		VertexShader* vertexShader,
		char** dVertexBuffer,
		TypeDescriptorEnum* dVertexTypeDescriptor,
		VaryingStore** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		ifloat4* dPosBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
		const auto numAttrs = deviceConstants->attributeCount;
		const auto numVaryings = deviceConstants->varyingCount;
		const void** vertexInputPtrs = new const void* [numAttrs];
		VaryingStore** varyingOutputPtrs = new VaryingStore * [numVaryings];
		for (int i = 0; i < numAttrs; i++) {
			vertexInputPtrs[i] = devGetBufferAddress(dVertexBuffer[i], dVertexTypeDescriptor[i], globalInvoIdx);
		}
		for (int i = 0; i < numVaryings; i++) {
			varyingOutputPtrs[i] = dVaryingBuffer[i] + globalInvoIdx;
		}
		vertexShader->execute(vertexInputPtrs, &dPosBuffer[globalInvoIdx], varyingOutputPtrs);
		delete[] vertexInputPtrs;
		delete[] varyingOutputPtrs;
	}

	IFRIT_KERNEL void geometryProcessingKernel(
		ifloat4* dPosBuffer,
		int* dIndexBuffer,
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
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
		if(!devTriangleCull(v1, v2, v3)) return;
		int startingIdx;
		int fw = devTriangleHomogeneousClip(primId, v1, v2, v3, dAssembledTriangles, dAssembledTriangleCount, &startingIdx);
		if (fw == -1) {
			return;
		}
		for(int i = startingIdx;i<startingIdx+fw;i++) {
			auto& atri = dAssembledTriangles[threadIdx.x][i];
			irect2Df bbox;
			if(!devTriangleSimpleClip(atri.v1, atri.v2, atri.v3, bbox)) continue;
			devExecuteBinner(i, atri, bbox, dRasterQueue, dRasterQueueCount, dCoverQueue, dCoverQueueCount, deviceConstants);
		}
	}

	IFRIT_KERNEL void tilingRasterizationKernel(
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto tileIdxX = blockIdx.x * blockDim.x + threadIdx.x;
		const auto tileIdxY = blockIdx.y * blockDim.y + threadIdx.y;
		const auto tileId = tileIdxY * deviceConstants->tileBlocksX + tileIdxX;
		const auto rastCandidates = dRasterQueueCount[tileId];
		
		const auto dispatchBlocks = (rastCandidates >> 5) + ((rastCandidates & 0x1F) != 0);
		tilingRasterizationChildKernel CU_KARG2(dispatchBlocks, 32) (tileIdxX, tileIdxY, rastCandidates,
			dAssembledTriangles, dAssembledTriangleCount, dRasterQueue, dRasterQueueCount, dCoverQueue,
			dCoverQueueCount, deviceConstants);

	}

	IFRIT_KERNEL void tilingRasterizationChildKernel(
		uint32_t tileIdX,
		uint32_t tileIdY,
		uint32_t totalBound,
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
		const auto tileId = tileIdY * deviceConstants->tileBlocksX + tileIdX;
		if (globalInvocation > totalBound)return;
		const uint32_t pixelStX = deviceConstants->frameBufferWidth * tileIdX / deviceConstants->tileBlocksX;
		const uint32_t pixelEdX = deviceConstants->frameBufferWidth * (tileIdX + 1) / deviceConstants->tileBlocksX;
		const uint32_t pixelStY = deviceConstants->frameBufferHeight * tileIdY / deviceConstants->tileBlocksX;
		const uint32_t pixelEdY = deviceConstants->frameBufferHeight * (tileIdY + 1) / deviceConstants->tileBlocksX;
		const auto primitiveSrcThread = dRasterQueue[tileId][globalInvocation].clippedTriangle.workerId;
		const auto primitiveSrcId = dRasterQueue[tileId][globalInvocation].clippedTriangle.primId;
		const auto& atri = dAssembledTriangles[primitiveSrcThread][primitiveSrcId];

		float tileMinX = 1.0f * tileIdX / deviceConstants->tileBlocksX;
		float tileMinY = 1.0f * tileIdY / deviceConstants->tileBlocksX;
		float tileMaxX = 1.0f * (tileIdX + 1) / deviceConstants->tileBlocksX;
		float tileMaxY = 1.0f * (tileIdY + 1) / deviceConstants->tileBlocksX;

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = atri.e1;
		edgeCoefs[1] = atri.e2;
		edgeCoefs[2] = atri.e3;

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

		// Decomp into Sub Blocks
		for (int i = deviceConstants->subtileBlocksX * deviceConstants->subtileBlocksX - 1; i >= 0; --i) {
			int criteriaTR = 0;
			int criteriaTA = 0;

			auto subTileIX = i % deviceConstants->subtileBlocksX;
			auto subTileIY = i / deviceConstants->subtileBlocksX;
			auto subTileTX = (tileIdX * deviceConstants->subtileBlocksX + subTileIX);
			auto subTileTY = (tileIdY * deviceConstants->subtileBlocksX + subTileIY);

			const int wp = (deviceConstants->subtileBlocksX * deviceConstants->tileBlocksX);
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
				nprop.tile = { subTileIX,subTileIY };
				nprop.clippedTriangle = dRasterQueue[tileId][globalInvocation].clippedTriangle;
				auto proposalInsIdx = atomicAdd(&dCoverQueueCount[tileId], 1);
				dCoverQueue[tileId][proposalInsIdx] = nprop;
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
							auto proposalInsIdx = atomicAdd(&dCoverQueueCount[tileId], 1);
							dCoverQueue[tileId][proposalInsIdx] = nprop;
						}
					}
				}
			}
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

		const auto tileX = pixelX * deviceConstants->tileBlocksX / deviceConstants->frameBufferWidth;
		const auto tileY = pixelY * deviceConstants->tileBlocksX / deviceConstants->frameBufferHeight;
		const auto tileId = tileY * deviceConstants->tileBlocksX + tileX;

		const auto subTileX = pixelX * deviceConstants->tileBlocksX * deviceConstants->subtileBlocksX / deviceConstants->frameBufferWidth;
		const auto subTileY = pixelY * deviceConstants->tileBlocksX * deviceConstants->subtileBlocksX / deviceConstants->frameBufferWidth;
		const auto subTileXInb = subTileX - tileX * deviceConstants->subtileBlocksX;
		const auto subTileYInb = subTileY - tileY * deviceConstants->subtileBlocksX;

		//TODO: Time consuming, looping to find related primitives

		VaryingStore* interpolatedVaryings = new VaryingStore[deviceConstants->varyingCount];
		ifloat4 colorOutputSingle;

		for (int i = dCoverQueueCount[tileId]; i >= 0; i--) {
			const auto& tileProposal = dCoverQueue[tileId][i];
			if (tileProposal.level == TileRasterLevel::PIXEL &&
				(tileProposal.tile.x != pixelX || tileProposal.tile.y != pixelY)) {
				continue;
			}
			else if (tileProposal.level == TileRasterLevel::BLOCK &&
				(tileProposal.tile.x != subTileXInb || tileProposal.tile.y != subTileYInb)) {
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


	IFRIT_KERNEL void testingKernel() {
		printf("Hello World\n");
	}
}


namespace  Ifrit::Engine::TileRaster::CUDA::Invocation {
	void testingKernelWrapper() {
		Impl::testingKernel CU_KARG2(4,4) ();
		cudaDeviceSynchronize();
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

	void invokeCudaRendering(
		char** hVertexBuffer,
		TypeDescriptorEnum* hVertexTypeDescriptor,
		TypeDescriptorEnum* hVaryingTypeDescriptor,
		int* hIndexBuffer,
		VertexShader* vertexShader,
		FragmentShader* fragmentShader,
		ifloat4** hColorBuffer,
		float** hDepthBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {
		// Host To Device Copy
		char** hdVertexBuffer;
		hdVertexBuffer = new char * [deviceConstants->attributeCount];
		for (int i = 0; i < deviceConstants->attributeCount; i++) {
			cudaMalloc(&hdVertexBuffer[i], deviceConstants->vertexCount * invokeCudaRenderingGetTpSize(hVertexTypeDescriptor[i]));
			cudaMemcpy(hdVertexBuffer[i], hVertexBuffer[i], deviceConstants->vertexCount * invokeCudaRenderingGetTpSize(hVertexTypeDescriptor[i]), cudaMemcpyHostToDevice);
		}
		char** dVertexBuffer;
		cudaMalloc(&dVertexBuffer, deviceConstants->attributeCount * sizeof(char*));
		cudaMemcpy(dVertexBuffer, hdVertexBuffer, deviceConstants->attributeCount * sizeof(char*), cudaMemcpyHostToDevice);

		int* dIndexBuffer;
		cudaMalloc(&dIndexBuffer, deviceConstants->indexCount * sizeof(int));
		cudaMemcpy(dIndexBuffer, hIndexBuffer, deviceConstants->indexCount * sizeof(int), cudaMemcpyHostToDevice);

		TypeDescriptorEnum* dVertexTypeDescriptor;
		cudaMalloc(&dVertexTypeDescriptor, deviceConstants->attributeCount * sizeof(TypeDescriptorEnum));
		cudaMemcpy(dVertexTypeDescriptor, hVertexTypeDescriptor, deviceConstants->attributeCount * sizeof(TypeDescriptorEnum), cudaMemcpyHostToDevice);

		TypeDescriptorEnum* dVaryingTypeDescriptor;
		cudaMalloc(&dVaryingTypeDescriptor, deviceConstants->varyingCount * sizeof(TypeDescriptorEnum));
		cudaMemcpy(dVaryingTypeDescriptor, hVaryingTypeDescriptor, deviceConstants->varyingCount * sizeof(TypeDescriptorEnum), cudaMemcpyHostToDevice);

		// Compute


		// Free Memory
		cudaFree(dVertexTypeDescriptor);
		cudaFree(dVaryingTypeDescriptor);
		cudaFree(dIndexBuffer);
		cudaFree(dVertexBuffer);
		for (int i = 0; i < deviceConstants->attributeCount; i++) {
			cudaFree(hdVertexBuffer[i]);
		}
		delete[] hdVertexBuffer;
	}
}