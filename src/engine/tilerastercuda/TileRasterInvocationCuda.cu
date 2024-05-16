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
					auto proposalId = atomicAdd(&dCoverQueueCount[workerId], 1);
					dCoverQueue[workerId][proposalId] = proposal;
				}
				else {
					TileBinProposal proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.clippedTriangle = { workerId,primitiveId };
					auto proposalId = atomicAdd(&dRasterQueueCount[workerId], 1);
					dRasterQueue[workerId][proposalId] = proposal;
				}
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
		char** dVaryingBuffer,
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
			varyingOutputPtrs[i] = reinterpret_cast<VaryingStore*>(devGetBufferAddress(dVaryingBuffer[i], dVaryingTypeDescriptor[i], globalInvoIdx));
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
}