#include "engine/tileraster/TileRasterWorker.h"
#include "engine/base/Shaders.h"
#include "math/VectorOps.h"
#include "math/simd/SimdVectors.h"

using namespace Ifrit::Math;
using namespace Ifrit::Math::SIMD;

namespace Ifrit::Engine::TileRaster {
	constexpr auto TOTAL_THREADS = TileRasterContext::numThreads + 1;
	TileRasterWorker::TileRasterWorker(uint32_t workerId, std::shared_ptr<TileRasterRenderer> renderer, std::shared_ptr<TileRasterContext> context) {
		this->workerId = workerId;
		this->rendererReference = renderer.get();
		this->context = context;
	}
	void TileRasterWorker::release() {
		status.store(TileRasterStage::TERMINATING);
		if (execWorker->joinable()) {
			execWorker->join();
		}
	}
	void TileRasterWorker::run() IFRIT_AP_NOTHROW {
		while (true) {
			const auto& curStatus = status.load();
			if (curStatus == TileRasterStage::CREATED || curStatus == TileRasterStage::COMPLETED) {
				std::this_thread::yield();
				continue;
			}
			else if(curStatus == TileRasterStage::TERMINATING){
				return;
			}
			else if(curStatus == TileRasterStage::DRAWCALL_START){
				drawCall();
			}
			else if (curStatus == TileRasterStage::DRAWCALL_START_CLEAR) {
				drawCallWithClear();
			}
			
		}
	}
	void TileRasterWorker::drawCallWithClear() IFRIT_AP_NOTHROW {
		context->frameBuffer->getColorAttachment(0)->clearImageZeroMultiThread(workerId, TOTAL_THREADS);
		context->frameBuffer->getDepthAttachment()->clearImageMultithread(255, workerId, TOTAL_THREADS);
		drawCall();
	}
	void TileRasterWorker::drawCall() IFRIT_AP_NOTHROW {
		auto rawRenderer = rendererReference;
		auto totalTiles = context->numTilesX * context->numTilesY;
		context->assembledTriangles[workerId].clear();
		for (int j = 0; j < totalTiles; j++) {
			context->rasterizerQueue[workerId][j].clear();
			context->coverQueue[workerId][j].clear();
		}
		vertexProcessing(rawRenderer);
		rawRenderer->statusTransitionBarrier2(TileRasterStage::VERTEX_SHADING_SYNC, TileRasterStage::GEOMETRY_PROCESSING);
		geometryProcessing(rawRenderer);
		rawRenderer->statusTransitionBarrier2(TileRasterStage::GEOMETRY_PROCESSING_SYNC, TileRasterStage::RASTERIZATION);
		rasterization(rawRenderer);
		if (context->optForceDeterministic) {
			rawRenderer->statusTransitionBarrier2(TileRasterStage::RASTERIZATION_SYNC, TileRasterStage::SORTING);
			sortOrderProcessing(rawRenderer);
			rawRenderer->statusTransitionBarrier2(TileRasterStage::SORTING_SYNC, TileRasterStage::FRAGMENT_SHADING);
		}else{
			rawRenderer->statusTransitionBarrier2(TileRasterStage::RASTERIZATION_SYNC, TileRasterStage::FRAGMENT_SHADING);
		}
		fragmentProcessing(rawRenderer);
	}
	uint32_t TileRasterWorker::triangleHomogeneousClip(const int primitiveId, vfloat4 v1, vfloat4 v2, vfloat4 v3) IFRIT_AP_NOTHROW {
		
		constexpr uint32_t clipIts = 2;
		const vfloat4 clipCriteria[2] = {
			vfloat4(0,0,0,1),
			vfloat4(0,0,0,0)
		};
		const vfloat4 clipNormal[2] = {
			vfloat4(0,0,0,-1),
			vfloat4(0,0,-1,0)
		};

		struct TileRasterClipVertex {
			vfloat4 barycenter;
			vfloat4 pos;
		};

		TileRasterClipVertex ret[2][9];
		uint32_t retCnt[2] = { 0,3 };
		ret[1][0] = { vfloat4(1,0,0,0),v1 };
		ret[1][1] = { vfloat4(0,1,0,0),v2 };
		ret[1][2] = { vfloat4(0,0,1,0),v3 };
		int clipTimes = 0;
		for (int i = 0; i < clipIts; i++) {
			vfloat4 outNormal = clipNormal[i];
			vfloat4 refPoint = clipCriteria[i];
			const auto cIdx = i & 1, cRIdx = 1 - (i & 1);
			retCnt[cIdx] = 0;
			const auto psize = retCnt[cRIdx];
			if (psize == 0) {
				return 0;
			}
			auto pc = ret[cRIdx][0];
			auto npc = dot(pc.pos, outNormal);
			for (int j = 0; j < psize; j++) {
				const auto& pn = ret[cRIdx][(j + 1) % psize];
				auto npn = dot(pn.pos, outNormal);
				if (npc * npn < 0) {
					vfloat4 dir = pn.pos - pc.pos;
					float numo = pc.pos.w - pc.pos.x * refPoint.x - pc.pos.y * refPoint.y - pc.pos.z * refPoint.z;
					float deno = dir.x * refPoint.x + dir.y * refPoint.y + dir.z * refPoint.z - dir.w;
					float t = (-refPoint.w + numo) / deno;
					vfloat4 intersection = pc.pos + dir * t;
					vfloat4 barycenter = lerp(pc.barycenter, pn.barycenter, t);

					TileRasterClipVertex newp;
					newp.barycenter = barycenter;
					newp.pos = intersection;
					ret[cIdx][retCnt[cIdx]++] = (newp);
				}
				if (npn < EPS) {
					ret[cIdx][retCnt[cIdx]++] = pn;
				}
				pc = pn;
				npc = npn;
			}
			if (retCnt[cIdx] < 3) {
				return 0;
			}
		}
		const auto clipOdd = clipTimes & 1;
		for (int i = 0; i < retCnt[clipOdd]; i++) {
			auto pd = ret[clipOdd][i].pos.w;
			ret[clipOdd][i].pos /= pd;
			ret[clipOdd][i].pos.w = pd;
		}
		for (int i = 0; i < retCnt[clipOdd] - 2; i++) {
			AssembledTriangleProposal atri;
			vfloat4 tv1, tv2, tv3;
			vfloat4 tb1, tb2, tb3;
			tb1 = ret[clipOdd][0].barycenter;
			tb2 = ret[clipOdd][i + 1].barycenter;
			tb3 = ret[clipOdd][i + 2].barycenter;
			tv1 = ret[clipOdd][0].pos;
			tv2 = ret[clipOdd][i + 1].pos;
			tv3 = ret[clipOdd][i + 2].pos;

			atri.bx = { tb1.x, tb2.x, tb3.x };
			atri.by = { tb1.y, tb2.y, tb3.y };

			auto edgeFunctionSimdVec = [](const vfloat4& a, const vfloat4& b, const vfloat4& c) {
				return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
			};

			const float ar = 1 / edgeFunctionSimdVec(tv1,tv2, tv3);
			const float sV2V1y = tv2.y - tv1.y;
			const float sV2V1x = tv1.x - tv2.x;
			const float sV3V2y = tv3.y - tv2.y;
			const float sV3V2x = tv2.x - tv3.x;
			const float sV1V3y = tv1.y - tv3.y;
			const float sV1V3x = tv3.x - tv1.x;

			const auto csInvX = 1.0f / context->frameBuffer->getColorAttachment(0)->getWidth();
			const auto csInvY = 1.0f / context->frameBuffer->getColorAttachment(0)->getHeight();

			atri.f3 = { sV2V1y * ar * csInvX * 2.0f, sV2V1x * ar * csInvY * 2.0f,(-tv1.x * sV2V1y - tv1.y * sV2V1x - sV2V1y - sV2V1x) * ar };
			atri.f1 = { sV3V2y * ar * csInvX * 2.0f, sV3V2x * ar * csInvY * 2.0f,(-tv2.x * sV3V2y - tv2.y * sV3V2x - sV3V2y - sV3V2x) * ar };
			atri.f2 = { sV1V3y * ar * csInvX * 2.0f, sV1V3x * ar * csInvY * 2.0f,(-tv3.x * sV1V3y - tv3.y * sV1V3x - sV1V3y - sV1V3x) * ar };


			atri.e1 = { 2.0f * sV2V1y,  2.0f * sV2V1x,  tv2.x * tv1.y - tv1.x * tv2.y - sV2V1y - sV2V1x - EPS };
			atri.e2 = { 2.0f * sV3V2y,  2.0f * sV3V2x,  tv3.x * tv2.y - tv2.x * tv3.y - sV3V2y - sV3V2x - EPS };
			atri.e3 = { 2.0f * sV1V3y,  2.0f * sV1V3x,  tv1.x * tv3.y - tv3.x * tv1.y - sV1V3y - sV1V3x - EPS };

			atri.originalPrimitive = primitiveId;
			// Precision might matter here
			atri.vw = vfloat3(1.0f / tv1.w, 1.0f / tv2.w, 1.0f / tv3.w);
			atri.vz = vfloat3(tv1.z, tv2.z, tv3.z);
			if (!triangleCulling(tv1, tv2, tv3)) {
				continue;
			}
			irect2Df bbox;
			if (!triangleFrustumClip(tv1, tv2, tv3, bbox)) {
				continue;
			}
			auto xid = context->assembledTriangles[workerId].size();
			context->assembledTriangles[workerId].emplace_back(std::move(atri));
			executeBinner(xid, atri, bbox);
		}
		return  retCnt[clipOdd] - 2;
	}
	bool TileRasterWorker::triangleFrustumClip(Ifrit::Math::SIMD::vfloat4 v1, Ifrit::Math::SIMD::vfloat4 v2, Ifrit::Math::SIMD::vfloat4 v3, irect2Df& bbox) IFRIT_AP_NOTHROW {
		auto bMin = min(v1, min(v2, v3));
		auto bMax = max(v1, max(v2, v3));
		if (bMax.z < 0.0) return false;
		if (bMin.z > 1.0) return false;
		if (bMax.x < -1.0) return false;
		if (bMin.x > 1.0) return false;
		if (bMax.y < -1.0) return false;
		if (bMin.y > 1.0) return false;
		bbox.x = bMin.x;
		bbox.y = bMin.y;
		bbox.w = bMax.x - bMin.x;
		bbox.h = bMax.y - bMin.y;
		return true;
	}
	bool TileRasterWorker::triangleCulling(vfloat4 v1, vfloat4 v2, vfloat4 v3) IFRIT_AP_NOTHROW {
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
	void TileRasterWorker::executeBinner(const int primitiveId, const AssembledTriangleProposal& atp, irect2Df bbox) IFRIT_AP_NOTHROW {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		float minx = bbox.x * 0.5f + 0.5f;
		float miny = bbox.y * 0.5f + 0.5f;
		float maxx = (bbox.x + bbox.w) * 0.5f + 0.5f;
		float maxy = (bbox.y + bbox.h) * 0.5f + 0.5f;

		auto frameBufferWidth = context->frameBuffer->getWidth();
		auto frameBufferHeight = context->frameBuffer->getHeight();


		int tileMinx = std::max(0, (int)(minx * frameBufferWidth / context->tileWidth));
		int tileMiny = std::max(0, (int)(miny * frameBufferHeight / context->tileWidth));
		int tileMaxx = (int)(maxx * frameBufferWidth / context->tileWidth);
		int tileMaxy = (int)(maxy * frameBufferWidth / context->tileWidth);

		ifloat3 edgeCoefs[3];
		edgeCoefs[0] = atp.e1;
		edgeCoefs[1] = atp.e2;
		edgeCoefs[2] = atp.e3;

		ifloat3 tileCoords[4];

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		getAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);


		const float tileSizeX = 1.0f * context->tileWidth / frameBufferWidth;
		const float tileSizeY = 1.0f * context->tileWidth / frameBufferHeight;
		for (int y = tileMiny; y <= tileMaxy; y++) {
			for (int x = tileMinx; x <= tileMaxx; x++) {
				tileCoords[VLT] = { x * tileSizeX, y * tileSizeY, 1.0 };
				tileCoords[VLB] = { x * tileSizeX, (y + 1) * tileSizeY, 1.0 };
				tileCoords[VRB] = { (x + 1) * tileSizeX, (y + 1) * tileSizeY, 1.0 };
				tileCoords[VRT] = { (x + 1) * tileSizeX, y * tileSizeY, 1.0 };

				int criteriaTR = 0;
				int criteriaTA = 0;
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoords[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTR[i]].y + edgeCoefs[i].z;
					float criteriaTALocal = edgeCoefs[i].x * tileCoords[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTA[i]].y + edgeCoefs[i].z;
					if (criteriaTRLocal < 0) criteriaTR += 1;
					if (criteriaTALocal < 0) criteriaTA += 1;
				}
				if (criteriaTR != 3)continue;
				if (criteriaTA == 3) {
					TileBinProposal proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.clippedTriangle = { workerId,primitiveId };
					context->coverQueue[workerId][getTileID(x, y)].push_back(proposal);
				}
				else {
					TileBinProposal proposal;
					proposal.level = TileRasterLevel::TILE;
					proposal.clippedTriangle = { workerId,primitiveId };
					context->rasterizerQueue[workerId][getTileID(x, y)].push_back(proposal);
				}
			}
		}
	}
	void TileRasterWorker::vertexProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW {
		status.store(TileRasterStage::VERTEX_SHADING, std::memory_order::relaxed);
		std::vector<vfloat4*> outVaryings(context->varyingDescriptor->getVaryingCounts());
		std::vector<const void*> inVertex(context->vertexBuffer->getAttributeCount());
		auto vsEntry = context->threadSafeVS[workerId];
		const auto vxCount = context->vertexBuffer->getVertexCount();
		for (int j = workerId; j < vxCount; j += TOTAL_THREADS) {
			auto pos = &context->vertexShaderResult->getPositionBuffer()[j];
			getVaryingsAddr(j, outVaryings);
			getVertexAttributes(j, inVertex);
			vsEntry->execute(inVertex.data(), pos, (ifloat4*const*)outVaryings.data());
		}
		status.store(TileRasterStage::VERTEX_SHADING_SYNC, std::memory_order::relaxed);
	}

	void TileRasterWorker::geometryProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW {
		auto posBuffer = context->vertexShaderResult->getPositionBuffer();
		generatedTriangle.clear();
		int genTris = 0, ixBufSize = context->indexBufferSize;
		for (int j = workerId * context->vertexStride; j < ixBufSize; j += TOTAL_THREADS * context->vertexStride) {
			int id0 = (context->indexBuffer)[j];
			int id1 = (context->indexBuffer)[j + 1];
			int id2 = (context->indexBuffer)[j + 2];

			if (context->frontface == TileRasterFrontFace::COUNTER_CLOCKWISE) {
				std::swap(id0, id2);
			}
			vfloat4 v1 = toSimdVector(posBuffer[id0]);
			vfloat4 v2 = toSimdVector(posBuffer[id1]); 
			vfloat4 v3 = toSimdVector(posBuffer[id2]);

			const auto prim = j / context->vertexStride;
			triangleHomogeneousClip(prim, v1, v2, v3);
		}
		status.store(TileRasterStage::GEOMETRY_PROCESSING_SYNC, std::memory_order::relaxed);
	}

	void TileRasterWorker::rasterization(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		auto curTile = 0;
		auto frameBufferWidth = context->frameBuffer->getWidth();
		auto frameBufferHeight = context->frameBuffer->getHeight();
		auto rdTiles = 0;
#ifdef IFRIT_USE_SIMD_128
		__m128 wfx128 = _mm_set1_ps(1.0f * context->subtileBlockWidth / frameBufferWidth);
		__m128 wfy128 = _mm_set1_ps(1.0f * context->subtileBlockWidth / frameBufferHeight);
#endif
		while ((curTile = renderer->fetchUnresolvedTileRaster()) >=0) {
			rdTiles++;
			int tileIdX = curTile % context->numTilesX;
			int tileIdY = curTile / context->numTilesX;

			float tileMinX = 1.0f * tileIdX * context->tileWidth / frameBufferWidth;
			float tileMinY = 1.0f * tileIdY * context->tileWidth / frameBufferHeight;
			float tileMaxX = 1.0f * (tileIdX + 1) * context->tileWidth / frameBufferWidth;
			float tileMaxY = 1.0f * (tileIdY + 1) * context->tileWidth / frameBufferHeight;

			for (int T = TOTAL_THREADS - 1; T >= 0; T--) {
				for (int j = context->rasterizerQueue[T][curTile].size() - 1; j >= 0; j--) {
					const auto& proposal = context->rasterizerQueue[T][curTile][j];
					const auto& ptRef = context->assembledTriangles[proposal.clippedTriangle.workerId][proposal.clippedTriangle.primId];

					ifloat3 edgeCoefs[3];
					edgeCoefs[0] = ptRef.e1;
					edgeCoefs[1] = ptRef.e2;
					edgeCoefs[2] = ptRef.e3;

					int chosenCoordTR[3];
					int chosenCoordTA[3];
					getAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

					int leftBlock = 0;
					int rightBlock = context->numSubtilesPerTileX - 1;
					int topBlock = 0;
					int bottomBlock = context->numSubtilesPerTileX - 1;

#ifdef IFRIT_USE_SIMD_128

					__m128 tileMinX128 = _mm_set1_ps(tileMinX);
					__m128 tileMinY128 = _mm_set1_ps(tileMinY);
					__m128 tileMaxX128 = _mm_set1_ps(tileMaxX);
					__m128 tileMaxY128 = _mm_set1_ps(tileMaxY);
					
					__m128 edgeCoefs128X[3], edgeCoefs128Y[3], edgeCoefs128Z[3];
					for (int k = 0; k < 3; k++) {
						edgeCoefs128X[k] = _mm_set1_ps(edgeCoefs[k].x);
						edgeCoefs128Y[k] = _mm_set1_ps(edgeCoefs[k].y);
						edgeCoefs128Z[k] = _mm_set1_ps(-edgeCoefs[k].z); //NOTE HERE
					}

#ifdef IFRIT_USE_SIMD_256
					__m256 tileMinX256 = _mm256_set1_ps(tileMinX);
					__m256 tileMinY256 = _mm256_set1_ps(tileMinY);
					__m256 tileMaxX256 = _mm256_set1_ps(tileMaxX);
					__m256 tileMaxY256 = _mm256_set1_ps(tileMaxY);
					//__m256 wp256 = _mm256_set1_ps(context->subtileBlocksX * context->tileBlocksX);
					__m256 frameBufferWidth256 = _mm256_set1_ps(frameBufferWidth);
					__m256 frameBufferHeight256 = _mm256_set1_ps(frameBufferHeight);

					__m256 edgeCoefs256X[3], edgeCoefs256Y[3], edgeCoefs256Z[3];
					for (int k = 0; k < 3; k++) {
						edgeCoefs256X[k] = _mm256_set1_ps(edgeCoefs[k].x * frameBufferHeight);
						edgeCoefs256Y[k] = _mm256_set1_ps(edgeCoefs[k].y * frameBufferWidth);
						edgeCoefs256Z[k] = _mm256_set1_ps((-edgeCoefs[k].z) * frameBufferHeight * frameBufferWidth);//NOTE HERE
					}
#endif

					TileBinProposal npropPixel;
					npropPixel.level = TileRasterLevel::PIXEL;
					npropPixel.clippedTriangle = proposal.clippedTriangle;

					TileBinProposal npropBlock;
					npropBlock.level = TileRasterLevel::BLOCK;
					npropBlock.clippedTriangle = proposal.clippedTriangle;

					TileBinProposal  npropPixel128;
					npropPixel128.level = TileRasterLevel::PIXEL_PACK2X2;
					npropPixel128.clippedTriangle = proposal.clippedTriangle;

					TileBinProposal  npropPixel256;
					npropPixel256.level = TileRasterLevel::PIXEL_PACK4X2;
					npropPixel256.clippedTriangle = proposal.clippedTriangle;

					auto& coverQueue = context->coverQueue[workerId][getTileID(tileIdX, tileIdY)];

					for (int x = leftBlock; x <= rightBlock; x += 2) {
						for (int y = topBlock; y <= bottomBlock; y += 2) {
							__m128i x128 = _mm_setr_epi32(x + 0, x + 1, x + 0, x + 1);
							__m128i y128 = _mm_setr_epi32(y + 0, y + 0, y + 1, y + 1);
							__m128i criteriaTR128 = _mm_setzero_si128();
							__m128i criteriaTA128 = _mm_setzero_si128();
							__m128 x128f = _mm_cvtepi32_ps(x128);
							__m128 y128f = _mm_cvtepi32_ps(y128);
							__m128 subTileMinX128 = _mm_fmadd_ps(x128f, wfx128,tileMinX128);
							__m128 subTileMinY128 = _mm_fmadd_ps(y128f, wfy128,tileMinY128);
							__m128 subTileMaxX128 = _mm_add_ps(subTileMinX128, wfx128);
							__m128 subTileMaxY128 = _mm_add_ps(subTileMinY128, wfy128);

							__m128 tileCoordsX128[4], tileCoordsY128[4];
							tileCoordsX128[VLT] = subTileMinX128;
							tileCoordsY128[VLT] = subTileMinY128;
							tileCoordsX128[VLB] = subTileMinX128;
							tileCoordsY128[VLB] = subTileMaxY128;
							tileCoordsX128[VRT] = subTileMaxX128;
							tileCoordsY128[VRT] = subTileMinY128;
							tileCoordsX128[VRB] = subTileMaxX128;
							tileCoordsY128[VRB] = subTileMaxY128;


							__m128 criteriaLocalTR128[3], criteriaLocalTA128[3];
							for (int k = 0; k < 3; k++) {
								criteriaLocalTR128[k] = _mm_fmadd_ps(edgeCoefs128X[k], tileCoordsX128[chosenCoordTR[k]],_mm_mul_ps(edgeCoefs128Y[k], tileCoordsY128[chosenCoordTR[k]]));
								criteriaLocalTA128[k] = _mm_fmadd_ps(edgeCoefs128X[k], tileCoordsX128[chosenCoordTA[k]],_mm_mul_ps(edgeCoefs128Y[k], tileCoordsY128[chosenCoordTA[k]]));

								__m128i criteriaTRMask = _mm_castps_si128(_mm_cmplt_ps(criteriaLocalTR128[k], edgeCoefs128Z[k]));
								__m128i criteriaTAMask = _mm_castps_si128(_mm_cmplt_ps(criteriaLocalTA128[k], edgeCoefs128Z[k]));
								criteriaTR128 = _mm_add_epi32(criteriaTR128, criteriaTRMask);
								criteriaTA128 = _mm_add_epi32(criteriaTA128, criteriaTAMask);
							}

							int criteriaTR[4], criteriaTA[4];
							_mm_storeu_si128((__m128i*)criteriaTR, criteriaTR128);
							_mm_storeu_si128((__m128i*)criteriaTA, criteriaTA128);

							for (int i = 0; i < 4; i++) {
								const auto dwX = x + (i & 1);
								const auto dwY = y + (i >> 1);
								if (criteriaTR[i] != -3) {
									continue;
								}
								if (criteriaTA[i] == -3) {
									npropBlock.tile = { dwX, dwY };
									coverQueue.push_back(npropBlock);
								}
								else {
									const auto subtilesXPerTile = context->numSubtilesPerTileX;
									const auto stMX = tileIdX * subtilesXPerTile + dwX;
									const auto stMY = tileIdY * subtilesXPerTile + dwY;
									const int subTileMinX = stMX * context->subtileBlockWidth;
									const int subTileMinY = stMY * context->subtileBlockWidth;
									const int subTileMaxX = (stMX + 1) * context->subtileBlockWidth;
									const int subTileMaxY = (stMY + 1) * context->subtileBlockWidth;


#ifdef IFRIT_USE_SIMD_256
									for (int dx = subTileMinX; dx < subTileMaxX; dx += 4) {
										for (int dy = subTileMinY; dy < subTileMaxY; dy += 2) {
											__m256 dx256 = _mm256_setr_ps(dx + 0, dx + 1, dx + 0, dx + 1, dx + 2, dx + 3, dx + 2, dx + 3);
											__m256 dy256 = _mm256_setr_ps(dy + 0, dy + 0, dy + 1, dy + 1, dy + 0, dy + 0, dy + 1, dy + 1);
											__m256i accept256 = _mm256_setzero_si256();
											__m256 criteria256[3];

											for (int k = 0; k < 3; k++) {
												criteria256[k] = _mm256_fmadd_ps(edgeCoefs256X[k], dx256, _mm256_mul_ps(edgeCoefs256Y[k], dy256));
												auto acceptMask = _mm256_castps_si256(_mm256_cmp_ps(criteria256[k], edgeCoefs256Z[k], _CMP_LT_OS));
												accept256 = _mm256_add_epi32(accept256, acceptMask);
											}
											accept256 = _mm256_cmpeq_epi32(accept256, _mm256_set1_epi32(-3));
											if (_mm256_testc_si256(accept256, _mm256_set1_epi32(-1))) {
												// If All Accept
												npropPixel256.tile = { dx,dy };
												coverQueue.push_back(npropPixel256);
											}
											else {
												// Pack By 2
												__m128i accept128[2];
												_mm256_storeu_si256((__m256i*)accept128, accept256);
												for (int di = 0; di < 2; di++) {
													auto pv = dx + ((di & 1) << 1);
													if (_mm_testc_si128(accept128[di], _mm_set1_epi32(-1))) {
														npropPixel128.tile = { pv, dy };
														coverQueue.push_back(npropPixel128);
													}
													else {
														int accept[4];
														_mm_storeu_si128((__m128i*)accept, accept128[di]);

														for (int ddi = 0; ddi < 4; ddi++) {
															const auto pvx = pv + (ddi & 1);
															const auto pvy = dy + (ddi >> 1);
															if (accept[ddi] == -1) {
																npropPixel.tile = { pvx,pvy };
																coverQueue.push_back(npropPixel);
															}
														}
													}
												}
											}

										}
									}
#else								
									for (int dx = subTileMinX; dx < subTileMaxX; dx += 2) {
										for (int dy = subTileMinY; dy < subTileMaxY; dy += 2) {
											__m128 dx128 = _mm_setr_ps(dx + 0, dx + 1, dx + 0, dx + 1);
											__m128 dy128 = _mm_setr_ps(dy + 0, dy + 0, dy + 1, dy + 1);
											__m128 ndcX128 = _mm_div_ps(dx128, _mm_set1_ps(frameBufferWidth));
											__m128 ndcY128 = _mm_div_ps(dy128, _mm_set1_ps(frameBufferHeight));
											__m128i accept128 = _mm_setzero_si128();
											__m128 criteria128[3];
											for (int k = 0; k < 3; k++) {
												criteria128[k] = _mm_add_ps(_mm_mul_ps(edgeCoefs128X[k], ndcX128), _mm_mul_ps(edgeCoefs128Y[k], ndcY128));
												__m128i acceptMask = _mm_castps_si128(_mm_cmplt_ps(criteria128[k], edgeCoefs128Z[k]));
												accept128 = _mm_add_epi32(accept128, acceptMask);
											}
											accept128 = _mm_cmpeq_epi32(accept128, _mm_set1_epi32(-3));

											if (_mm_testc_si128(accept128, _mm_set1_epi32(-1))) {
												// If All Accept
												npropPixel128.tile = { dx,dy };
												npropPixel128.clippedTriangle = proposal.clippedTriangle;
												context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(npropPixel128);
											}
											else {
												int accept[4];
												_mm_storeu_si128((__m128i*)accept, accept128);
												for (int di = 0; di < 4; di++) {
													if (accept[di] == -1 && dx + di % 2 < frameBufferWidth && dy + di / 2 < frameBufferHeight) {
														npropPixel.tile = { dx + di % 2, dy + di / 2 };
														npropPixel.clippedTriangle = proposal.clippedTriangle;
														context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(npropPixel);
													}
												}
											}
										}
									}
#endif
								}
							}
						}
					}

#else
					auto totalSubtiles = context->numSubtilesPerTileX * context->numSubtilesPerTileX;
					for (int vx = 0; vx < totalSubtiles; vx++) {
						int x = vx % context->numSubtilesPerTileX;
						int y = vx / context->numSubtilesPerTileX;
						int criteriaTR = 0;
						int criteriaTA = 0;

						//float wp = (context->subtileBlocksX * context->tileBlocksX);
						float subTileMinX = tileMinX + 1.0f * x * context->subtileBlockWidth / frameBufferWidth;
						float subTileMinY = tileMinY + 1.0f * y * context->subtileBlockWidth / frameBufferHeight;
						float subTileMaxX = tileMinX + 1.0f * (x + 1) * context->subtileBlockWidth / frameBufferWidth;
						float subTileMaxY = tileMinY + 1.0f * (y + 1) * context->subtileBlockWidth / frameBufferHeight;

						int subTilePixelX = (tileIdX * context->numSubtilesPerTileX + x) * context->subtileBlockWidth;
						int subTilePixelY = (tileIdY * context->numSubtilesPerTileX + y) * context->subtileBlockWidth;
						int subTilePixelX2 = (tileIdX * context->numSubtilesPerTileX + x + 1) * context->subtileBlockWidth;
						int subTilePixelY2 = (tileIdY * context->numSubtilesPerTileX + y + 1) * context->subtileBlockWidth;

						subTilePixelX2 = std::min(subTilePixelX2 * 1u, frameBufferWidth - 1);
						subTilePixelY2 = std::min(subTilePixelY2 * 1u, frameBufferHeight - 1);

						ifloat3 tileCoords[4];
						tileCoords[VLT] = { subTileMinX, subTileMinY, 1.0 };
						tileCoords[VLB] = { subTileMinX, subTileMaxY, 1.0 };
						tileCoords[VRB] = { subTileMaxX, subTileMaxY, 1.0 };
						tileCoords[VRT] = { subTileMaxX, subTileMinY, 1.0 };

						for (int k = 0; k < 3; k++) {
							float criteriaTRLocal = edgeCoefs[k].x * tileCoords[chosenCoordTR[k]].x + edgeCoefs[k].y * tileCoords[chosenCoordTR[k]].y + edgeCoefs[k].z;
							float criteriaTALocal = edgeCoefs[k].x * tileCoords[chosenCoordTA[k]].x + edgeCoefs[k].y * tileCoords[chosenCoordTA[k]].y + edgeCoefs[k].z;
							if (criteriaTRLocal < -EPS2) criteriaTR += 1;
							if (criteriaTALocal < EPS2) criteriaTA += 1;
						}


						if (criteriaTR != 3) {
							continue;
						}
						if (criteriaTA == 3) {
							TileBinProposal nprop;
							nprop.level = TileRasterLevel::BLOCK;
							nprop.tile = { x,y };
							nprop.clippedTriangle = proposal.clippedTriangle;
							context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(nprop);
						}
						else {
							for (int dx = subTilePixelX; dx < subTilePixelX2; dx++) {
								for (int dy = subTilePixelY; dy < subTilePixelY2; dy++) {
									float ndcX = 1.0f * dx / frameBufferWidth;
									float ndcY = 1.0f * dy / frameBufferHeight;
									int accept = 0;
									for (int i = 0; i < 3; i++) {
										float criteria = edgeCoefs[i].x * ndcX + edgeCoefs[i].y * ndcY + edgeCoefs[i].z;
										accept += criteria < EPS2;
									}
									if (accept == 3) {
										TileBinProposal nprop;
										nprop.level = TileRasterLevel::PIXEL;
										nprop.tile = { dx,dy };
										nprop.clippedTriangle = proposal.clippedTriangle;
										context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(nprop);
									}
								}
							}
						}

					}
#endif
				}
			}
		}

		status.store(TileRasterStage::RASTERIZATION_SYNC, std::memory_order::relaxed);
	}

	void TileRasterWorker::sortOrderProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW {
		auto curTile = 0;
		while ((curTile = renderer->fetchUnresolvedTileSort()) >=0) {
			std::vector<int> numSpaces(TOTAL_THREADS);
			int preSum = 0;
			for (int i = 0; i < TOTAL_THREADS; i++) {
				numSpaces[i] = preSum;
				preSum += context->coverQueue[i][curTile].size();
			}
			context->sortedCoverQueue[curTile].resize(preSum);
			for (int i = 0; i < TOTAL_THREADS; i++) {
				std::copy(context->coverQueue[i][curTile].begin(), context->coverQueue[i][curTile].end(),
					context->sortedCoverQueue[curTile].begin() + numSpaces[i]);
			}
			auto sortCompareOp = [&](const TileBinProposal& a, const TileBinProposal& b) {
				auto aw = a.clippedTriangle.workerId;
				auto ap = a.clippedTriangle.primId;
				auto ao = context->assembledTriangles[aw][ap].originalPrimitive;
				auto bw = b.clippedTriangle.workerId;
				auto bp = b.clippedTriangle.primId;
				auto bo = context->assembledTriangles[bw][bp].originalPrimitive;
				return ao < bo;
				};
			std::sort(context->sortedCoverQueue[curTile].begin(), context->sortedCoverQueue[curTile].end(), sortCompareOp);
		}
		status.store(TileRasterStage::SORTING_SYNC, std::memory_order::relaxed);
	}

	void TileRasterWorker::fragmentProcessing(TileRasterRenderer* renderer) IFRIT_AP_NOTHROW {
		auto curTile = 0;
		const auto frameBufferWidth = context->frameBuffer->getWidth();
		const auto frameBufferHeight = context->frameBuffer->getHeight();
		const auto varyingCnts = context->varyingDescriptor->getVaryingCounts();
		interpolatedVaryings.reserve(varyingCnts);
		interpolatedVaryings.resize(varyingCnts);
		interpolatedVaryingsAddr.reserve(varyingCnts);
		interpolatedVaryingsAddr.resize(varyingCnts);

		for (int i = interpolatedVaryingsAddr.size() - 1; i >= 0; i--) {
			interpolatedVaryingsAddr[i] = &interpolatedVaryings[i];
		}
		PixelShadingFuncArgs pxArgs;
		pxArgs.colorAttachment0 = context->frameBuffer->getColorAttachment(0);
		pxArgs.depthAttachmentPtr = context->frameBuffer->getDepthAttachment();
		pxArgs.varyingCounts = varyingCnts;
		pxArgs.indexBufferPtr = (context->indexBuffer);
		
		while ((curTile = renderer->fetchUnresolvedTileFragmentShading())>=0) {

			auto proposalProcessFunc = [&]<bool tpAlphaBlendEnable,IfritCompareOp tpDepthFunc>(TileBinProposal& proposal) {
				const auto& triProposal = context->assembledTriangles[proposal.clippedTriangle.workerId][proposal.clippedTriangle.primId];
				if (proposal.level == TileRasterLevel::PIXEL) IFRIT_BRANCH_LIKELY{
					pixelShading<tpAlphaBlendEnable,tpDepthFunc>(triProposal, proposal.tile.x, proposal.tile.y,pxArgs);
				}
				else if (proposal.level == TileRasterLevel::PIXEL_PACK4X2) {
#ifdef IFRIT_USE_SIMD_128
#ifdef IFRIT_USE_SIMD_256
					pixelShadingSIMD256<tpAlphaBlendEnable,tpDepthFunc>(triProposal, proposal.tile.x, proposal.tile.y, pxArgs);
#else
					for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 3u, frameBufferWidth - 1); dx += 2) {
						for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
							pixelShadingSIMD128<tpAlphaBlendEnable,tpDepthFunc>(triProposal, dx, dy, pxArgs);
						}
					}
#endif
#else
					for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 3u, frameBufferWidth - 1); dx++) {
						for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
							pixelShading<tpAlphaBlendEnable, tpDepthFunc>(triProposal, dx, dy, pxArgs);
						}
					}
#endif
				}
				else if (proposal.level == TileRasterLevel::PIXEL_PACK2X2) {
#ifdef IFRIT_USE_SIMD_128
					pixelShadingSIMD128<tpAlphaBlendEnable, tpDepthFunc>(triProposal, proposal.tile.x, proposal.tile.y, pxArgs);
#else
					for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 1u, frameBufferWidth - 1); dx++) {
						for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
							pixelShading<tpAlphaBlendEnable, tpDepthFunc>(triProposal, dx, dy, pxArgs);
						}
					}
#endif
				}
				else if (proposal.level == TileRasterLevel::TILE) {
					auto curTileX = curTile % context->numTilesX;
					auto curTileY = curTile / context->numTilesX;

					auto curTileX2 = (curTileX + 1) * context->tileWidth;
					auto curTileY2 = (curTileY + 1) * context->tileWidth;
					curTileX = curTileX * context->tileWidth;
					curTileY = curTileY * context->tileWidth;
					curTileX2 = std::min(curTileX2, (int)frameBufferWidth);
					curTileY2 = std::min(curTileY2, (int)frameBufferHeight);

					for (int dx = curTileX; dx < curTileX2; dx++) {
						for (int dy = curTileY; dy < curTileY2; dy++) {
							pixelShading<tpAlphaBlendEnable, tpDepthFunc>(triProposal, dx, dy, pxArgs);
						}
					}
				}
				else if (proposal.level == TileRasterLevel::BLOCK) {
					auto curTileX = curTile % context->numTilesX;
					auto curTileY = curTile / context->numTilesX;
					auto subtileXPerTile = context->numSubtilesPerTileX;
					auto subTilePixelX = (curTileX * subtileXPerTile + proposal.tile.x) * context->subtileBlockWidth;
					auto subTilePixelY = (curTileY * subtileXPerTile + proposal.tile.y) * context->subtileBlockWidth;
					auto subTilePixelX2 = (curTileX * subtileXPerTile + proposal.tile.x + 1) * context->subtileBlockWidth;
					auto subTilePixelY2 = (curTileY * subtileXPerTile + proposal.tile.y + 1) * context->subtileBlockWidth;
					subTilePixelX2 = std::min(subTilePixelX2, (int)frameBufferWidth);
					subTilePixelY2 = std::min(subTilePixelY2, (int)frameBufferHeight);
					for (int dx = subTilePixelX; dx < subTilePixelX2; dx++) {
						for (int dy = subTilePixelY; dy < subTilePixelY2; dy++) {
							pixelShading<tpAlphaBlendEnable, tpDepthFunc>(triProposal, dx, dy, pxArgs);
						}
					}
				}
				};
			// End of lambda func
			if (context->optForceDeterministic) {
				auto iterFunc = [&]<bool tpAlphaBlendEnable,IfritCompareOp tpDepthFunc>() {
					for (auto& proposal: context->sortedCoverQueue[curTile]) {
						proposalProcessFunc.operator()<tpAlphaBlendEnable, tpDepthFunc>(proposal);
					}
				};
#define IF_DECLPS_ITERFUNC_0(tpAlphaBlendEnable,tpDepthFunc) iterFunc.operator()<tpAlphaBlendEnable,tpDepthFunc>();
#define IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,tpDepthFunc) if(context->depthFunc == tpDepthFunc) IF_DECLPS_ITERFUNC_0(tpAlphaBlendEnable,tpDepthFunc)

#define IF_DECLPS_ITERFUNC(tpAlphaBlendEnable) \
	IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,IF_COMPARE_OP_ALWAYS); \
	IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,IF_COMPARE_OP_EQUAL); \
	IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,IF_COMPARE_OP_GREATER); \
	IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,IF_COMPARE_OP_GREATER_OR_EQUAL); \
	IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,IF_COMPARE_OP_LESS); \
	IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,IF_COMPARE_OP_LESS_OR_EQUAL); \
	IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,IF_COMPARE_OP_NEVER); \
	IF_DECLPS_ITERFUNC_0_BRANCH(tpAlphaBlendEnable,IF_COMPARE_OP_NOT_EQUAL); \

				if (context->blendState.blendEnable){
					IF_DECLPS_ITERFUNC(true);
				}
				else{
					IF_DECLPS_ITERFUNC(false);
				}
			}
			else {
				auto iterFunc = [&]<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc>() {
					for (int i = TOTAL_THREADS - 1; i >= 0; i--) {
						for (int j = context->coverQueue[i][curTile].size() - 1; j >= 0; j--) {
							auto& proposal = context->coverQueue[i][curTile][j];
							proposalProcessFunc.operator()<tpAlphaBlendEnable, tpDepthFunc >(proposal);
						}
					}
				};
				if (context->blendState.blendEnable) {
					IF_DECLPS_ITERFUNC(true);
				}
				else {
					IF_DECLPS_ITERFUNC(false);
				}
			}
#undef IF_DECLPS_ITERFUNC_0_BRANCH
#undef IF_DECLPS_ITERFUNC_0
#undef IF_DECLPS_ITERFUNC

		}
		status.store(TileRasterStage::FRAGMENT_SHADING_SYNC, std::memory_order::relaxed);
	}

	void TileRasterWorker::threadStart() {
		execWorker = std::make_unique<std::thread>(&TileRasterWorker::run, this);
		//execWorker->detach();
	}

	void TileRasterWorker::getVertexAttributes(const int id, std::vector<const void*>& out) IFRIT_AP_NOTHROW {
		for (int i = 0; i < context->vertexBuffer->getAttributeCount(); i++) {
			auto desc = context->vertexBuffer->getAttributeDescriptor(i);
			if (desc.type == TypeDescriptorEnum::IFTP_FLOAT4) IFRIT_BRANCH_LIKELY{
				out[i] = (context->vertexBuffer->getValuePtr<ifloat4>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::IFTP_FLOAT3) {
				out[i] = (context->vertexBuffer->getValuePtr<ifloat3>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::IFTP_FLOAT2) {
				out[i] = (context->vertexBuffer->getValuePtr<ifloat2>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::IFTP_FLOAT1) {
				out[i] = (context->vertexBuffer->getValuePtr<float>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::IFTP_INT1) {
				out[i] = (context->vertexBuffer->getValuePtr<int>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::IFTP_INT2) {
				out[i] = (context->vertexBuffer->getValuePtr<iint2>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::IFTP_INT3) {
				out[i] = (context->vertexBuffer->getValuePtr<iint3>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::IFTP_INT4) {
				out[i] = (context->vertexBuffer->getValuePtr<iint4>(id, i));
			}
			else IFRIT_BRANCH_UNLIKELY{
				ifritError("Unsupported Type");
			}
		}
	}
	void TileRasterWorker::getVaryingsAddr(const int id, std::vector<vfloat4*>& out) IFRIT_AP_NOTHROW {
		for (int i = 0; i < context->varyingDescriptor->getVaryingCounts(); i++) {
			auto desc = context->vertexShaderResult->getVaryingDescriptor(i);
			out[i] = &context->vertexShaderResult->getVaryingBuffer(i)[id];
		}
	}
	template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc>
	void TileRasterWorker::pixelShading(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW {

		auto& depthAttachment = (*(args.depthAttachmentPtr))(dx, dy, 0);
		int idx = atp.originalPrimitive * context->vertexStride;

		vfloat4 pDxDyVec = vfloat4(dx, dy, 1.0f, 0.0f);
		vfloat3 zVec = atp.vz;
		vfloat3 wVec = atp.vw;

		float bary[3];
		float interpolatedDepth;
		bary[0] = dot(pDxDyVec, atp.f1);
		bary[1] = dot(pDxDyVec, atp.f2);
		bary[2] = dot(pDxDyVec, atp.f3);
		vfloat3 baryVec = vfloat3(bary[0], bary[1], bary[2]);
		interpolatedDepth = dot(zVec, baryVec);

		// Depth Test
		if constexpr (tpDepthFunc == IF_COMPARE_OP_ALWAYS) {
			
		}
		if constexpr (tpDepthFunc == IF_COMPARE_OP_EQUAL) {
			if (interpolatedDepth != depthAttachment) return;
		}
		if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER) {
		}
		if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER_OR_EQUAL) {
			if (interpolatedDepth < depthAttachment) return;
		}
		if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS) {
			if (interpolatedDepth >= depthAttachment) return;
		}
		if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS_OR_EQUAL) {
			if (interpolatedDepth > depthAttachment) return;
		}
		if constexpr (tpDepthFunc == IF_COMPARE_OP_NEVER) {
			return;
		}
		if constexpr (tpDepthFunc == IF_COMPARE_OP_NOT_EQUAL) {
			if (interpolatedDepth == depthAttachment) return;
		}

		baryVec *= wVec;
		float zCorr = 1.0f / hsum(baryVec);
		baryVec *= zCorr;
		vfloat3 atpBx = atp.bx;
		vfloat3 atpBy = atp.by;
		float desiredBary[3];
		desiredBary[0] = dot(baryVec, atpBx);
		desiredBary[1] = dot(baryVec, atpBy);
		desiredBary[2] = 1.0f - desiredBary[0] - desiredBary[1];

		const int* const addr = args.indexBufferPtr + idx;
		const auto vSize = args.varyingCounts;
		for (int i = 0; i < vSize; i++) {
			auto va = context->vertexShaderResult->getVaryingBuffer(i);
			auto& dest = interpolatedVaryings[i];
			const auto& tmp0 = (va[addr[0]]);
			const auto& tmp1 = (va[addr[1]]);
			const auto& tmp2 = (va[addr[2]]);
			vfloat4 destVec =tmp0 * desiredBary[0];
			destVec = fma(tmp1, desiredBary[1], destVec);
			dest = fma(tmp2, desiredBary[2], destVec);
		}
		// Fragment Shader
		auto& psEntry = context->threadSafeFS[workerId];
		psEntry->execute(interpolatedVaryings.data(), colorOutput.data(), &interpolatedDepth);

		if constexpr (tpAlphaBlendEnable) {
			auto dstRgba = args.colorAttachment0->getPixelRGBAUnsafe(dx, dy);
			auto srcRgba = colorOutput[0];
			const auto& blendParam = context->blendColorCoefs;
			const auto& blendParamAlpha = context->blendAlphaCoefs;
			auto mxSrcX = (blendParam.s.x + blendParam.s.y * (1 - srcRgba.w) + blendParam.s.z * (1 - dstRgba[3])) * (1 - blendParam.s.w);
			auto mxDstX = (blendParam.d.x + blendParam.d.y * (1 - srcRgba.w) + blendParam.d.z * (1 - dstRgba[3])) * (1 - blendParam.d.w);
			auto mxSrcA = (blendParamAlpha.s.x + blendParamAlpha.s.y * (1 - srcRgba.w) + blendParamAlpha.s.z * (1 - dstRgba[3])) * (1 - blendParamAlpha.s.w);
			auto mxDstA = (blendParamAlpha.d.x + blendParamAlpha.d.y * (1 - srcRgba.w) + blendParamAlpha.d.z * (1 - dstRgba[3])) * (1 - blendParamAlpha.d.w);
			
			ifloat4 mixRgba;
			mixRgba.x = dstRgba[0] * mxDstX + srcRgba.x * mxSrcX;
			mixRgba.y = dstRgba[1] * mxDstX + srcRgba.y * mxSrcX;
			mixRgba.z = dstRgba[2] * mxDstX + srcRgba.z * mxSrcX;
			mixRgba.w = dstRgba[3] * mxDstA + srcRgba.w * mxSrcA;
#ifdef IFRIT_USE_SIMD_128
			args.colorAttachment0->fillPixelRGBA128ps(dx, dy, _mm_loadu_ps((float*)(&mixRgba)));
#else
			args.colorAttachment0->fillPixelRGBA(dx, dy, mixRgba.x, mixRgba.y, mixRgba.z, mixRgba.w);
#endif
		}
		else {
#ifdef IFRIT_USE_SIMD_128
			args.colorAttachment0->fillPixelRGBA128ps(dx, dy, _mm_loadu_ps((float*)(&colorOutput[0])));
#else
			args.colorAttachment0->fillPixelRGBA(dx, dy, colorOutput[0].x, colorOutput[0].y, colorOutput[0].z, colorOutput[0].w);
#endif
		}
		// Depth Write
		depthAttachment = interpolatedDepth;
	}

	template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc>
	void TileRasterWorker::pixelShadingSIMD128(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW {
#ifndef IFRIT_USE_SIMD_128
		ifritError("SIMD 128 not enabled");
#else
		const auto fbWidth = context->frameWidth;
		const auto fbHeight = context->frameHeight;

		auto& depthAttachment = *args.depthAttachmentPtr;

		int idx = atp.originalPrimitive * context->vertexStride;

		__m128 posZ[3], posW[3];
		posZ[0] = _mm_set1_ps(atp.vz.x);
		posZ[1] = _mm_set1_ps(atp.vz.y);
		posZ[2] = _mm_set1_ps(atp.vz.z);
		posW[0] = _mm_set1_ps(atp.vw.x);
		posW[1] = _mm_set1_ps(atp.vw.y);
		posW[2] = _mm_set1_ps(atp.vw.z);

		__m128i dx128i = _mm_setr_epi32(dx + 0, dx + 1, dx + 0, dx + 1);
		__m128i dy128i = _mm_setr_epi32(dy + 0, dy + 0, dy + 1, dy + 1);

		__m128 attachmentWidth = _mm_set1_ps(context->frameBuffer->getWidth());
		__m128 attachmentHeight = _mm_set1_ps(context->frameBuffer->getHeight());

		__m128 pDx = _mm_cvtepi32_ps(dx128i);
		__m128 pDy = _mm_cvtepi32_ps(dy128i);

		__m128 bary[3];
		bary[0] = _mm_fmadd_ps(_mm_set1_ps(atp.f1.y), pDy, _mm_set1_ps(atp.f1.z));
		bary[0] = _mm_fmadd_ps(_mm_set1_ps(atp.f1.x), pDx, bary[0]);
		bary[1] = _mm_fmadd_ps(_mm_set1_ps(atp.f2.y), pDy, _mm_set1_ps(atp.f2.z));
		bary[1] = _mm_fmadd_ps(_mm_set1_ps(atp.f2.x), pDx, bary[1]);
		bary[2] = _mm_fmadd_ps(_mm_set1_ps(atp.f3.y), pDy, _mm_set1_ps(atp.f3.z));
		bary[2] = _mm_fmadd_ps(_mm_set1_ps(atp.f3.x), pDx, bary[2]);

		__m128 baryDiv[3];
		for (int i = 0; i < 3; i++) {
			baryDiv[i] = _mm_mul_ps(bary[i], posW[i]);
		}
		__m128 barySum = _mm_add_ps(_mm_add_ps(baryDiv[0], baryDiv[1]), baryDiv[2]);
		__m128 zCorr = _mm_rcp_ps(barySum);

		__m128 interpolatedDepth128 = _mm_fmadd_ps(bary[0], posZ[0], _mm_mul_ps(bary[1], posZ[1]));
		interpolatedDepth128 = _mm_fmadd_ps(bary[2], posZ[2], interpolatedDepth128);

		float interpolatedDepth[4] = { 0 };
		float bary32[3][4];
		float zCorr32[4];
		_mm_storeu_ps(interpolatedDepth, interpolatedDepth128);
		_mm_storeu_ps(zCorr32, zCorr);
		for (int i = 0; i < 3; i++) {
			_mm_storeu_ps(bary32[i], baryDiv[i]);
		}

		const int* const addr = args.indexBufferPtr + idx;
		const auto varyCounts = args.varyingCounts;
		vfloat3 atpBxVec = atp.bx;
		vfloat3 atpByVec = atp.by;
		for (int i = 0; i < 4; i++) {
			//Depth Test
			int x = dx + (i & 1);
			int y = dy + (i >> 1);
			if (x >= fbWidth || y >= fbHeight){
				continue;
			}

			const auto depthAttachment2 = depthAttachment(x, y, 0);
			if constexpr (tpDepthFunc == IF_COMPARE_OP_ALWAYS) {

			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_EQUAL) {
				if (interpolatedDepth[i] != depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER) {
				if (interpolatedDepth[i] <= depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER_OR_EQUAL) {
				if (interpolatedDepth[i] < depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS) {
				if (interpolatedDepth[i] >= depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS_OR_EQUAL) {
				if (interpolatedDepth[i] > depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_NEVER) {
				return;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_NOT_EQUAL) {
				if (interpolatedDepth[i] == depthAttachment2) continue;
			}

			//float barytmp[3] = { bary32[0][i] * zCorr32[i],bary32[1][i] * zCorr32[i],bary32[2][i] * zCorr32[i] };
			vfloat3 bary32Vec = vfloat3(bary32[0][i], bary32[1][i], bary32[2][i]) * zCorr32[i];

			float desiredBary[3];
			desiredBary[0] = dot(bary32Vec, atpBxVec);
			desiredBary[1] = dot(bary32Vec, atpByVec);
			desiredBary[2] = 1.0f - desiredBary[0] - desiredBary[1];

			for (int k = 0; k < varyCounts; k++) {
				auto va = context->vertexShaderResult->getVaryingBuffer(k);
				auto& dest = interpolatedVaryings[k];
				const auto& tmp0 = (va[addr[0]]);
				const auto& tmp1 = (va[addr[1]]);
				const auto& tmp2 = (va[addr[2]]);
				vfloat4 destVec = tmp0 * desiredBary[0];
				destVec = fma(tmp1, desiredBary[1], destVec);
				dest = fma(tmp2, desiredBary[2], destVec);
			}

			// Fragment Shader
			auto& psEntry = context->threadSafeFS[workerId];
			psEntry->execute(interpolatedVaryings.data(), colorOutput.data(), &interpolatedDepth[i]);
			
			if constexpr(tpAlphaBlendEnable) {
				auto dstRgba = args.colorAttachment0->getPixelRGBAUnsafe(x, y);
				auto srcRgba = colorOutput[0];
				const auto& blendParam = context->blendColorCoefs;
				const auto& blendParamAlpha = context->blendAlphaCoefs;
				auto mxSrcX = (blendParam.s.x + blendParam.s.y * (1 - srcRgba.w) + blendParam.s.z * (1 - dstRgba[3])) * (1 - blendParam.s.w);
				auto mxDstX = (blendParam.d.x + blendParam.d.y * (1 - srcRgba.w) + blendParam.d.z * (1 - dstRgba[3])) * (1 - blendParam.d.w);
				auto mxSrcA = (blendParamAlpha.s.x + blendParamAlpha.s.y * (1 - srcRgba.w) + blendParamAlpha.s.z * (1 - dstRgba[3])) * (1 - blendParamAlpha.s.w);
				auto mxDstA = (blendParamAlpha.d.x + blendParamAlpha.d.y * (1 - srcRgba.w) + blendParamAlpha.d.z * (1 - dstRgba[3])) * (1 - blendParamAlpha.d.w);

				ifloat4 mixRgba;
				mixRgba.x = dstRgba[0] * mxDstX + srcRgba.x * mxSrcX;
				mixRgba.y = dstRgba[1] * mxDstX + srcRgba.y * mxSrcX;
				mixRgba.z = dstRgba[2] * mxDstX + srcRgba.z * mxSrcX;
				mixRgba.w = dstRgba[3] * mxDstA + srcRgba.w * mxSrcA;
				args.colorAttachment0->fillPixelRGBA128ps(x, y, _mm_loadu_ps((const float*)(&mixRgba)));
			}
			else {
				args.colorAttachment0->fillPixelRGBA128ps(x, y, _mm_loadu_ps((const float*)(&colorOutput[0])));
			}
			// Depth Write
			depthAttachment(x, y, 0) = interpolatedDepth[i];
		}
#endif
	}

	template<bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc>
	void TileRasterWorker::pixelShadingSIMD256(const AssembledTriangleProposal& atp, const int dx, const int dy,const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW {
#ifndef IFRIT_USE_SIMD_256
		ifritError("SIMD 256 (AVX2) not enabled");
#else
		int idx = atp.originalPrimitive * context->vertexStride;
		const auto fbWidth = context->frameWidth;
		const auto fbHeight = context->frameHeight;

		__m256 posZ[3], posW[3];
		posZ[0] = _mm256_set1_ps(atp.vz.x);
		posZ[1] = _mm256_set1_ps(atp.vz.y);
		posZ[2] = _mm256_set1_ps(atp.vz.z);
		posW[0] = _mm256_set1_ps(atp.vw.x);
		posW[1] = _mm256_set1_ps(atp.vw.y);
		posW[2] = _mm256_set1_ps(atp.vw.z);

		__m256i dx256i = _mm256_setr_epi32(dx + 0, dx + 1, dx + 0, dx + 1, dx + 2, dx + 3, dx + 2, dx + 3);
		__m256i dy256i = _mm256_setr_epi32(dy + 0, dy + 0, dy + 1, dy + 1, dy + 0, dy + 0, dy + 1, dy + 1);

		__m256 attachmentWidth = _mm256_set1_ps(fbWidth);
		__m256 attachmentHeight = _mm256_set1_ps(fbHeight);

		__m256 pDx = _mm256_cvtepi32_ps(dx256i);
		__m256 pDy = _mm256_cvtepi32_ps(dy256i);

		__m256 bary[3];
		bary[0] = _mm256_fmadd_ps(_mm256_set1_ps(atp.f1.y), pDy, _mm256_set1_ps(atp.f1.z));
		bary[0] = _mm256_fmadd_ps(_mm256_set1_ps(atp.f1.x), pDx, bary[0]);
		bary[1] = _mm256_fmadd_ps(_mm256_set1_ps(atp.f2.y), pDy, _mm256_set1_ps(atp.f2.z));
		bary[1] = _mm256_fmadd_ps(_mm256_set1_ps(atp.f2.x), pDx, bary[1]);
		bary[2] = _mm256_fmadd_ps(_mm256_set1_ps(atp.f3.y), pDy, _mm256_set1_ps(atp.f3.z));
		bary[2] = _mm256_fmadd_ps(_mm256_set1_ps(atp.f3.x), pDx, bary[2]);
			
		__m256 baryDiv[3];
		for (int i = 0; i < 3; i++) {
			baryDiv[i] = _mm256_mul_ps(bary[i], posW[i]);
		}

		__m256 baryDivWSum = _mm256_add_ps(_mm256_add_ps(baryDiv[0], baryDiv[1]), baryDiv[2]);
		__m256 zCorr = _mm256_rcp_ps(baryDivWSum);

		__m256 interpolatedDepth256 = _mm256_fmadd_ps(bary[0], posZ[0], _mm256_fmadd_ps(bary[1], posZ[1], _mm256_mul_ps(bary[2], posZ[2])));
	
		float interpolatedDepth[8] = { 0 };
		float bary32[3][8];
		float zCorr32[8];
		_mm256_storeu_ps(interpolatedDepth, interpolatedDepth256);
		_mm256_storeu_ps(zCorr32, zCorr);
		for (int i = 0; i < 3; i++) {
			_mm256_storeu_ps(bary32[i], baryDiv[i]);
		}

		auto& depthAttachment = *args.depthAttachmentPtr;
		const int* const addr = args.indexBufferPtr + idx;
		vfloat3 atpBxVec = atp.bx;
		vfloat3 atpByVec = atp.by;
		for (int i = 0; i < 8; i++) {
			//Depth Test
			int x = dx + ((i & 1)) + ((i >> 2) << 1);
			int y = dy + ((i & 3) >> 1);
			if (x >= fbWidth || y >= fbHeight) IFRIT_BRANCH_UNLIKELY{
				continue;
			}
			const auto depthAttachment2 = depthAttachment(x, y, 0);
			if constexpr (tpDepthFunc == IF_COMPARE_OP_ALWAYS) {

			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_EQUAL) {
				if (interpolatedDepth[i] != depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER) {
				if (interpolatedDepth[i] <= depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER_OR_EQUAL) {
				if (interpolatedDepth[i] < depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS) {
				if (interpolatedDepth[i] >= depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS_OR_EQUAL) {
				if (interpolatedDepth[i] > depthAttachment2) continue;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_NEVER) {
				return;
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_NOT_EQUAL) {
				if (interpolatedDepth[i] == depthAttachment2) continue;
			}
			vfloat3 bary32Vec = vfloat3(bary32[0][i], bary32[1][i], bary32[2][i]) * zCorr32[i];
			float desiredBary[3];
			desiredBary[0] = dot(bary32Vec, atpBxVec);
			desiredBary[1] = dot(bary32Vec, atpByVec);
			desiredBary[2] = 1 - desiredBary[0] - desiredBary[1];
			for (int k = 0; k < args.varyingCounts; k++) {
				//interpolateVaryings(k, addr, desiredBary, interpolatedVaryings[k]);
				auto va = context->vertexShaderResult->getVaryingBuffer(k);
				auto& dest = interpolatedVaryings[k];
				const auto& tmp0 = (va[addr[0]]);
				const auto& tmp1 = (va[addr[1]]);
				const auto& tmp2 = (va[addr[2]]);
				vfloat4 destVec = tmp0 * desiredBary[0];
				destVec = fma(tmp1, desiredBary[1], destVec);
				dest = fma(tmp2, desiredBary[2], destVec);
			}

			// Fragment Shader
			auto& psEntry = context->threadSafeFS[workerId];
			psEntry->execute(interpolatedVaryings.data(), colorOutput.data(), &interpolatedDepth[i]);
			
			if constexpr (tpAlphaBlendEnable) {
				auto dstRgba = args.colorAttachment0->getPixelRGBAUnsafe(x, y);
				auto srcRgba = colorOutput[0];
				const auto& blendParam = context->blendColorCoefs;
				const auto& blendParamAlpha = context->blendAlphaCoefs;
				auto mxSrcX = (blendParam.s.x + blendParam.s.y * (1 - srcRgba.w) + blendParam.s.z * (1 - dstRgba[3])) * (1 - blendParam.s.w);
				auto mxDstX = (blendParam.d.x + blendParam.d.y * (1 - srcRgba.w) + blendParam.d.z * (1 - dstRgba[3])) * (1 - blendParam.d.w);
				auto mxSrcA = (blendParamAlpha.s.x + blendParamAlpha.s.y * (1 - srcRgba.w) + blendParamAlpha.s.z * (1 - dstRgba[3])) * (1 - blendParamAlpha.s.w);
				auto mxDstA = (blendParamAlpha.d.x + blendParamAlpha.d.y * (1 - srcRgba.w) + blendParamAlpha.d.z * (1 - dstRgba[3])) * (1 - blendParamAlpha.d.w);

				ifloat4 mixRgba;
				mixRgba.x = dstRgba[0] * mxDstX + srcRgba.x * mxSrcX;
				mixRgba.y = dstRgba[1] * mxDstX + srcRgba.y * mxSrcX;
				mixRgba.z = dstRgba[2] * mxDstX + srcRgba.z * mxSrcX;
				mixRgba.w = dstRgba[3] * mxDstA + srcRgba.w * mxSrcA;
				args.colorAttachment0->fillPixelRGBA128ps(x, y, _mm_loadu_ps((const float*)(&mixRgba)));
			}
			else {
				args.colorAttachment0->fillPixelRGBA128ps(x, y, _mm_loadu_ps((const float*)(&colorOutput[0])));
			}

			// Depth Write
			depthAttachment(x, y, 0) = interpolatedDepth[i];
		}
#endif
	}

	void TileRasterWorker::interpolateVaryings(int id, const int indices[3], const float barycentric[3], ifloat4& dest) IFRIT_AP_NOTHROW {
		auto va = context->vertexShaderResult->getVaryingBuffer(id);
		dest = { 0,0,0,0 };
		for (int j = 0; j < 3; j++) {
			dest.x += va[indices[j]].x * barycentric[j];
			dest.y += va[indices[j]].y * barycentric[j];
			dest.z += va[indices[j]].z * barycentric[j];
			dest.w += va[indices[j]].w * barycentric[j];
		}
	}

#define IF_DECLPS1(tpAlphaBlending,tpDepthFunc) \
	template void TileRasterWorker::pixelShading<tpAlphaBlending,tpDepthFunc>(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW; \
	template void TileRasterWorker::pixelShadingSIMD128<tpAlphaBlending,tpDepthFunc>(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW; \
	template void TileRasterWorker::pixelShadingSIMD256<tpAlphaBlending, tpDepthFunc>(const AssembledTriangleProposal& atp, const int dx, const int dy, const PixelShadingFuncArgs& args) IFRIT_AP_NOTHROW;

#define IF_DECLPS1_1(tpDepthFunc) IF_DECLPS1(true,tpDepthFunc);IF_DECLPS1(false,tpDepthFunc);

	IF_DECLPS1_1(IF_COMPARE_OP_ALWAYS);
	IF_DECLPS1_1(IF_COMPARE_OP_EQUAL);
	IF_DECLPS1_1(IF_COMPARE_OP_GREATER);
	IF_DECLPS1_1(IF_COMPARE_OP_GREATER_OR_EQUAL);
	IF_DECLPS1_1(IF_COMPARE_OP_LESS);
	IF_DECLPS1_1(IF_COMPARE_OP_LESS_OR_EQUAL);
	IF_DECLPS1_1(IF_COMPARE_OP_NEVER);
	IF_DECLPS1_1(IF_COMPARE_OP_NOT_EQUAL);

#undef IF_DECLPS1_1
#undef IF_DECLPS1
}