#include "engine/tileraster/TileRasterWorker.h"
#include "engine/base/FragmentShader.h"

namespace Ifrit::Engine::TileRaster {
	TileRasterWorker::TileRasterWorker(uint32_t workerId, std::shared_ptr<TileRasterRenderer> renderer, std::shared_ptr<TileRasterContext> context) {
		this->workerId = workerId;
		this->renderer = renderer;
		this->context = context;
	}
	void TileRasterWorker::run() {
		while (true) {
			if (status.load() == TileRasterStage::CREATED || status.load() == TileRasterStage::TERMINATED || activated.load()==false) {
				std::this_thread::yield();
				continue;
			}
			if (status.load() == TileRasterStage::VERTEX_SHADING) {
				vertexProcessing();
				activated.store(false);
			}
			if (status.load() == TileRasterStage::GEOMETRY_PROCESSING) {
				geometryProcessing();
				activated.store(false);
			}
			if (status.load() == TileRasterStage::RASTERIZATION) {
				rasterization();
				activated.store(false);
			}
			if (status.load() == TileRasterStage::FRAGMENT_SHADING) {
				fragmentProcessing();
				activated.store(false);
			}
		}


	}
	bool TileRasterWorker::triangleFrustumClip(float4 v1, float4 v2, float4 v3, rect2Df& bbox) {
		bool inside = true;
		float minx = std::min(v1.x, std::min(v2.x, v3.x));
		float miny = std::min(v1.y, std::min(v2.y, v3.y));
		float maxx = std::max(v1.x, std::max(v2.x, v3.x));
		float maxy = std::max(v1.y, std::max(v2.y, v3.y));
		float maxz = std::max(v1.z, std::max(v2.z, v3.z));
		float minz = std::min(v1.z, std::min(v2.z, v3.z));
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
	bool TileRasterWorker::triangleCulling(float4 v1, float4 v2, float4 v3) {
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
	void TileRasterWorker::executeBinner(const int primitiveId, float4 v1, float4 v2, float4 v3, rect2Df bbox) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		const float tileSize = 1.0 / context->tileBlocksX;
		float minx = bbox.x * 0.5 + 0.5;
		float miny = bbox.y * 0.5 + 0.5;
		float maxx = (bbox.x + bbox.w) * 0.5 + 0.5;
		float maxy = (bbox.y + bbox.h) * 0.5 + 0.5;

		int tileMinx = std::max(0, (int)(minx / tileSize));
		int tileMiny = std::max(0, (int)(miny / tileSize));
		int tileMaxx = std::min(context->tileBlocksX - 1, (int)(maxx / tileSize));
		int tileMaxy = std::min(context->tileBlocksX - 1, (int)(maxy / tileSize));

		float3 edgeCoefs[3] = { context->primitiveEdgeCoefs[primitiveId].coef[0],
	context->primitiveEdgeCoefs[primitiveId].coef[1],
	context->primitiveEdgeCoefs[primitiveId].coef[2] };

		float3 tileCoords[4];

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		auto frameBufferWidth = context->frameBuffer->getWidth();
		auto frameBufferHeight = context->frameBuffer->getHeight();
		getAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

		for (int y = tileMiny; y <= tileMaxy; y++) {
			for (int x = tileMinx; x <= tileMaxx; x++) {
				int criteriaTR = 0;
				int criteriaTA = 0;
				
				auto curTileX = x * frameBufferWidth / context->tileBlocksX;
				auto curTileY = y * frameBufferHeight / context->tileBlocksX;
				auto curTileX2 = (x + 1) * frameBufferWidth / context->tileBlocksX;
				auto curTileY2 = (y + 1) * frameBufferHeight / context->tileBlocksX;

				tileCoords[VLT] = { x * tileSize, y * tileSize, 1.0 };
				tileCoords[VLB] = { x * tileSize, (y + 1) * tileSize, 1.0 };
				tileCoords[VRB] = { (x + 1) * tileSize, (y + 1) * tileSize, 1.0 };
				tileCoords[VRT] = { (x + 1) * tileSize, y * tileSize, 1.0 };

				for (int i = 0; i < 4; i++) {
					tileCoords[i].x = tileCoords[i].x * 2 - 1;
					tileCoords[i].y = tileCoords[i].y * 2 - 1;
				}

				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoords[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTR[i]].y + edgeCoefs[i].z;
					float criteriaTALocal = edgeCoefs[i].x * tileCoords[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTA[i]].y + edgeCoefs[i].z;
					if (criteriaTRLocal < 0) criteriaTR += 1;
					if (criteriaTALocal < 0) criteriaTA += 1;
				}
				if (criteriaTR != 3)continue;
				if (criteriaTA == 3) {
					TileBinProposal proposal;
					proposal.allAccept = true;
					proposal.level = TileRasterLevel::TILE;
					proposal.bbox = bbox;
					proposal.primitiveId = primitiveId;
					context->coverQueue[workerId][getTileID(x, y)].push_back(proposal);
				}
				else {
					TileBinProposal proposal;
					proposal.allAccept = false;
					proposal.bbox = bbox;
					proposal.level = TileRasterLevel::TILE;
					proposal.primitiveId = primitiveId;
					context->rasterizerQueue[workerId][getTileID(x, y)].push_back(proposal);
				}
			}
		}
		
	}
	void TileRasterWorker::vertexProcessing() {
		perVertexVaryings.resize(context->vertexBuffer->getVertexCount() * context->vertexShader->getVaryingCounts());
		auto sk = context->vertexShader->getVaryingCounts();
		status.store(TileRasterStage::VERTEX_SHADING);
		std::vector<VaryingStore*> outVaryings(context->vertexShader->getVaryingCounts());
		std::vector<const void*> inVertex(context->vertexBuffer->getAttributeCount());
		for (int j = workerId; j < context->vertexBuffer->getVertexCount(); j += context->numThreads) {
			auto pos = &context->vertexShaderResult->getPositionBuffer()[j];
			getVaryingsAddr(j,outVaryings);
			getVertexAttributes(j,inVertex);
			context->vertexShader->execute(inVertex, *pos, outVaryings);
			pos->x /= pos->w;
			pos->y /= pos->w;
			pos->z /= pos->w;

			
		}
		status.store(TileRasterStage::VERTEX_SHADING_SYNC);
	}

	void TileRasterWorker::geometryProcessing() {
		auto posBuffer = context->vertexShaderResult->getPositionBuffer();
		for (int j = workerId * context->vertexStride; j < context->indexBuffer->size(); j += context->numThreads * context->vertexStride) {
			int id0 = (*context->indexBuffer)[j];
			int id1 = (*context->indexBuffer)[j + 1];
			int id2 = (*context->indexBuffer)[j + 2];

			if (context->frontface == TileRasterFrontFace::COUNTER_CLOCKWISE) {
				std::swap(id0, id2);
			}
			float4 v1 = posBuffer[id0];
			float4 v2 = posBuffer[id1];
			float4 v3 = posBuffer[id2];

			const auto prim = j / context->vertexStride;
			context->primitiveEdgeCoefs[prim].coef[0] = { v2.y - v1.y, v1.x - v2.x, v2.x * v1.y - v1.x * v2.y };
			context->primitiveEdgeCoefs[prim].coef[1] = { v3.y - v2.y, v2.x - v3.x, v3.x * v2.y - v2.x * v3.y };
			context->primitiveEdgeCoefs[prim].coef[2] = { v1.y - v3.y, v3.x - v1.x, v1.x * v3.y - v3.x * v1.y };

			for (int i = 0; i < 3; i++) {
				auto norm = std::max(abs(context->primitiveEdgeCoefs[prim].coef[i].x), abs(context->primitiveEdgeCoefs[prim].coef[i].y));
				context->primitiveEdgeCoefs[prim].coef[i].x /= norm;
				context->primitiveEdgeCoefs[prim].coef[i].y /= norm;
				context->primitiveEdgeCoefs[prim].coef[i].z /= norm;
			}

			rect2Df bbox;
			if (!triangleFrustumClip(v1, v2, v3, bbox)) {
				continue;
			}
			if (!triangleCulling(v1, v2, v3)) {
				continue;
			}
			executeBinner(prim, v1, v2, v3, bbox);
			context->primitiveMinZ[prim] = std::min(v1.z, std::min(v2.z, v3.z));
		}
		status.store(TileRasterStage::GEOMETRY_PROCESSING_SYNC);
	}

	void TileRasterWorker::rasterization(){
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		auto curTile = 0;
		auto frameBufferWidth = context->frameBuffer->getWidth();
		auto frameBufferHeight = context->frameBuffer->getHeight();
		auto rdTiles = 0;
		while ((curTile = renderer->fetchUnresolvedTileRaster()) != -1) {
			rdTiles++;
			int tileIdX = curTile % context->tileBlocksX;
			int tileIdY = curTile / context->tileBlocksX;

			float tileMinX = 1.0f * tileIdX / context->tileBlocksX;
			float tileMinY = 1.0f * tileIdY / context->tileBlocksX;
			float tileMaxX = 1.0f * (tileIdX + 1) / context->tileBlocksX;
			float tileMaxY = 1.0f * (tileIdY + 1) / context->tileBlocksX;


			for (int T = 0; T < context->numThreads; T++) {
				for (int j = 0; j < context->rasterizerQueue[T][curTile].size(); j++) {
					auto& proposal = context->rasterizerQueue[T][curTile][j];
					int idx0 = proposal.primitiveId * context->vertexStride;
					int idx1 = idx0 + 1;
					int idx2 = idx0 + 2;
					if (context->frontface == TileRasterFrontFace::COUNTER_CLOCKWISE) {
						std::swap(idx0, idx2);
					}

					//auto edgeCoefs = context->primitiveEdgeCoefs[proposal.primitiveId].coef;
					float3 edgeCoefs[3] = { context->primitiveEdgeCoefs[proposal.primitiveId].coef[0],
						context->primitiveEdgeCoefs[proposal.primitiveId].coef[1],
						context->primitiveEdgeCoefs[proposal.primitiveId].coef[2] };

					int chosenCoordTR[3];
					int chosenCoordTA[3];
					getAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

					rect2Df bbox = proposal.bbox;
					bbox.x = bbox.x * 0.5 + 0.5;
					bbox.y = bbox.y * 0.5 + 0.5;
					bbox.w = bbox.w * 0.5;
					bbox.h = bbox.h * 0.5;

					int leftBlock = 0;
					int rightBlock = context->subtileBlocksX - 1;
					int topBlock = 0;
					int bottomBlock = context->subtileBlocksX - 1;

#ifdef IFRIT_USE_SIMD_128

					__m128 tileMinX128 = _mm_set1_ps(tileMinX);
					__m128 tileMinY128 = _mm_set1_ps(tileMinY);
					__m128 tileMaxX128 = _mm_set1_ps(tileMaxX);
					__m128 tileMaxY128 = _mm_set1_ps(tileMaxY);
					__m128 wp128 = _mm_set1_ps(context->subtileBlocksX * context->tileBlocksX);

					__m128 edgeCoefs128X[3], edgeCoefs128Y[3], edgeCoefs128Z[3];
					for (int k = 0; k < 3; k++) {
						edgeCoefs128X[k] = _mm_set1_ps(edgeCoefs[k].x);
						edgeCoefs128Y[k] = _mm_set1_ps(edgeCoefs[k].y);
						edgeCoefs128Z[k] = _mm_set1_ps(edgeCoefs[k].z);
					}

#ifdef IFRIT_USE_SIMD_256
					__m256 tileMinX256 = _mm256_set1_ps(tileMinX);
					__m256 tileMinY256 = _mm256_set1_ps(tileMinY);
					__m256 tileMaxX256 = _mm256_set1_ps(tileMaxX);
					__m256 tileMaxY256 = _mm256_set1_ps(tileMaxY);
					__m256 wp256 = _mm256_set1_ps(context->subtileBlocksX * context->tileBlocksX);

					__m256 edgeCoefs256X[3], edgeCoefs256Y[3], edgeCoefs256Z[3];
					for (int k = 0; k < 3; k++) {
						edgeCoefs256X[k] = _mm256_set1_ps(edgeCoefs[k].x);
						edgeCoefs256Y[k] = _mm256_set1_ps(edgeCoefs[k].y);
						edgeCoefs256Z[k] = _mm256_set1_ps(edgeCoefs[k].z);
					}	
#endif

					TileBinProposal npropPixel;
					npropPixel.allAccept = true;
					npropPixel.level = TileRasterLevel::PIXEL;

					TileBinProposal  npropPixel128;
					npropPixel128.allAccept = true;
					npropPixel128.level = TileRasterLevel::PIXEL_PACK2X2;

					TileBinProposal  npropPixel256;
					npropPixel256.allAccept = true;
					npropPixel256.level = TileRasterLevel::PIXEL_PACK4X2;

					for (int x = leftBlock; x <= rightBlock; x += 2) {
						for (int y = topBlock; y <= bottomBlock; y += 2) {
							__m128i x128 = _mm_setr_epi32(x + 0, x + 1, x + 0, x + 1);
							__m128i y128 = _mm_setr_epi32(y + 0, y + 0, y + 1, y + 1);
							__m128i criteriaTR128 = _mm_setzero_si128();
							__m128i criteriaTA128 = _mm_setzero_si128();

							__m128 x128f = _mm_cvtepi32_ps(x128);
							__m128 y128f = _mm_cvtepi32_ps(y128);
							__m128 subTileMinX128 = _mm_add_ps(tileMinX128, _mm_div_ps(x128f, wp128));
							__m128 subTileMinY128 = _mm_add_ps(tileMinY128, _mm_div_ps(y128f, wp128));
							__m128 subTileMaxX128 = _mm_add_ps(tileMinX128, _mm_div_ps(_mm_add_ps(x128f, _mm_set1_ps(1.0f)), wp128));
							__m128 subTileMaxY128 = _mm_add_ps(tileMinY128, _mm_div_ps(_mm_add_ps(y128f, _mm_set1_ps(1.0f)), wp128));

							__m128 tileCoordsX128[4], tileCoordsY128[4];
							tileCoordsX128[VLT] = subTileMinX128;
							tileCoordsY128[VLT] = subTileMinY128;
							tileCoordsX128[VLB] = subTileMinX128;
							tileCoordsY128[VLB] = subTileMaxY128;
							tileCoordsX128[VRT] = subTileMaxX128;
							tileCoordsY128[VRT] = subTileMinY128;
							tileCoordsX128[VRB] = subTileMaxX128;
							tileCoordsY128[VRB] = subTileMaxY128;

							for (int k = 0; k < 4; k++) {
								tileCoordsX128[k] = _mm_sub_ps(_mm_mul_ps(tileCoordsX128[k], _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));
								tileCoordsY128[k] = _mm_sub_ps(_mm_mul_ps(tileCoordsY128[k], _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));
							}

							__m128 criteriaLocalTR128[3], criteriaLocalTA128[3];
							for (int k = 0; k < 3; k++) {
								criteriaLocalTR128[k] = _mm_add_ps(
									_mm_add_ps(_mm_mul_ps(edgeCoefs128X[k], tileCoordsX128[chosenCoordTR[k]]),
									_mm_mul_ps(edgeCoefs128Y[k], tileCoordsY128[chosenCoordTR[k]])), edgeCoefs128Z[k]);

								criteriaLocalTA128[k] = _mm_add_ps(
									_mm_add_ps(_mm_mul_ps(edgeCoefs128X[k], tileCoordsX128[chosenCoordTA[k]]),
									_mm_mul_ps(edgeCoefs128Y[k], tileCoordsY128[chosenCoordTA[k]])), edgeCoefs128Z[k]);	

								__m128i criteriaTRMask = _mm_castps_si128(_mm_cmplt_ps(criteriaLocalTR128[k], _mm_set1_ps(0)));
								__m128i criteriaTAMask = _mm_castps_si128(_mm_cmplt_ps(criteriaLocalTA128[k], _mm_set1_ps(0)));
								criteriaTRMask = _mm_sub_epi32(_mm_set1_epi32(0), criteriaTRMask);
								criteriaTAMask = _mm_sub_epi32(_mm_set1_epi32(0), criteriaTAMask);
								criteriaTR128 = _mm_add_epi32(criteriaTR128, criteriaTRMask);
								criteriaTA128 = _mm_add_epi32(criteriaTA128, criteriaTAMask);
							}


							int criteriaTR[4], criteriaTA[4];
							_mm_storeu_si128((__m128i*)criteriaTR, criteriaTR128);
							_mm_storeu_si128((__m128i*)criteriaTA, criteriaTA128);

							for (int i = 0; i < 4; i++) {
								auto dwX = x + i % 2;
								auto dwY = y + i / 2;
								if (criteriaTR[i] != 3 || (dwX > rightBlock || dwY > bottomBlock)) {
									continue;
								}
								
								if (criteriaTA[i] == 3) {
									TileBinProposal nprop;
									nprop.allAccept = true;
									nprop.level = TileRasterLevel::BLOCK;
									nprop.bbox = proposal.bbox;
									nprop.tile = { x + i % 2, y + i / 2 };
									nprop.primitiveId = proposal.primitiveId;
									context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(nprop);
								}
								else {
									float wp = (context->subtileBlocksX * context->tileBlocksX);
									int subTileMinX = (tileIdX * context->subtileBlocksX + (x + i % 2)) * frameBufferWidth / wp;
									int subTileMinY = (tileIdY * context->subtileBlocksX + (y + i / 2)) * frameBufferHeight / wp;
									int subTileMaxX = (tileIdX * context->subtileBlocksX + (x + i % 2) + 1) * frameBufferWidth / wp;
									int subTileMaxY = (tileIdY * context->subtileBlocksX + (y + i / 2) + 1) * frameBufferHeight / wp;
									subTileMaxX = std::min(subTileMaxX * 1u, frameBufferWidth - 1);
									subTileMaxY = std::min(subTileMaxY * 1u, frameBufferHeight - 1);


#ifdef IFRIT_USE_SIMD_256
									for (int dx = subTileMinX; dx <= subTileMaxX; dx += 4) {
										for (int dy = subTileMinY; dy <= subTileMaxY; dy += 2) {
											__m256i dx256 = _mm256_setr_epi32(dx + 0, dx + 1, dx + 0, dx + 1, dx + 2, dx + 3, dx + 2, dx + 3);
											__m256i dy256 = _mm256_setr_epi32(dy + 0, dy + 0, dy + 1, dy + 1, dy + 0, dy + 0, dy + 1, dy + 1);

											__m256 ndcX128 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_div_ps(_mm256_cvtepi32_ps(dx256), _mm256_set1_ps(frameBufferWidth))), _mm256_set1_ps(1.0f));
											__m256 ndcY128 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_div_ps(_mm256_cvtepi32_ps(dy256), _mm256_set1_ps(frameBufferHeight))), _mm256_set1_ps(1.0f));
											__m256i accept256 = _mm256_setzero_si256();
											__m256 criteria256[3];

											for (int k = 0; k < 3; k++) {
												criteria256[k] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(edgeCoefs256X[k], ndcX128), _mm256_mul_ps(edgeCoefs256Y[k], ndcY128)), edgeCoefs256Z[k]);
												auto acceptMask = _mm256_castps_si256(_mm256_cmp_ps(criteria256[k], _mm256_set1_ps(EPS), _CMP_LT_OS));
												acceptMask = _mm256_sub_epi32(_mm256_set1_epi32(0), acceptMask);
												accept256 = _mm256_add_epi32(accept256, acceptMask);
											}
											if (_mm256_testc_si256(_mm256_cmpeq_epi32(accept256, _mm256_set1_epi32(3)), _mm256_set1_epi32(-1))) {
												// If All Accept
												npropPixel256.bbox = proposal.bbox;
												npropPixel256.primitiveId = proposal.primitiveId;
												npropPixel256.tile = { dx,dy };
												context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(npropPixel256);
											}
											else {
												// Pack By 2
												__m128i accept128[2];
												_mm256_storeu_si256((__m256i*)accept128, accept256);
												for (int di = 0; di < 2; di++) {
													auto pv = dx + 2 * (di % 2);
													auto pw = dy;
													subTileMaxX = std::min(1u * subTileMaxX, frameBufferWidth-1);
													subTileMaxY = std::min(1u * subTileMaxY, frameBufferHeight-1);
													if (pv <= subTileMaxX && dy<= subTileMaxY &&
														_mm_movemask_epi8(_mm_cmpeq_epi32(accept128[di], _mm_set1_epi32(3))) == 0xFFFF) {
														npropPixel128.bbox = proposal.bbox;
														npropPixel128.primitiveId = proposal.primitiveId;
														npropPixel128.tile = { dx + 2 * (di % 2), dy  };
														context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(npropPixel128);
													}
													else {
														int accept[4];
														_mm_storeu_si128((__m128i*)accept, accept128[di]);
														
														for (int ddi = 0; ddi < 4; ddi++) {
															auto pvx = dx + ddi % 2 + 2 * (di % 2);
															auto pvy = dy + ddi / 2;
															if (pvx <= subTileMaxX && pvy <= subTileMaxY && accept[ddi] == 3) {
																npropPixel.bbox = proposal.bbox;
																npropPixel.primitiveId = proposal.primitiveId;
																npropPixel.tile = { dx + ddi % 2 + 2 * (di % 2), dy + ddi / 2};
																if (dx + ddi % 2 + 2 * (di % 2) == 1600) {
																	ifritLog1("??");
																}
																context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(npropPixel);
															}
														}
													}
												}
											}

										}
									}
#else
									for (int dx = subTileMinX; dx <= subTileMaxX; dx+=2) {
										for (int dy = subTileMinY; dy <= subTileMaxY; dy+=2) {
											__m128 dx128 = _mm_setr_ps(dx + 0, dx + 1, dx + 0, dx + 1);
											__m128 dy128 = _mm_setr_ps(dy + 0, dy + 0, dy + 1, dy + 1);
											__m128 ndcX128 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_div_ps(dx128, _mm_set1_ps(frameBufferWidth))), _mm_set1_ps(1.0f));
											__m128 ndcY128 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_div_ps(dy128, _mm_set1_ps(frameBufferHeight))), _mm_set1_ps(1.0f));
											__m128i accept128 = _mm_setzero_si128();
											__m128 criteria128[3];
											for (int k = 0; k < 3; k++) {
												criteria128[k] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(edgeCoefs128X[k], ndcX128),_mm_mul_ps(edgeCoefs128Y[k], ndcY128)),edgeCoefs128Z[k]);
												__m128i acceptMask = _mm_castps_si128(_mm_cmplt_ps(criteria128[k], _mm_set1_ps(EPS2)));
												acceptMask = _mm_sub_epi32(_mm_set1_epi32(0), acceptMask);
												accept128 = _mm_add_epi32(accept128, acceptMask);
											}
											
											if (_mm_movemask_epi8(_mm_cmpeq_epi32(accept128, _mm_set1_epi32(3))) == 0xFFFF) {
												// If All Accept
												npropPixel128.bbox = proposal.bbox;
												npropPixel128.primitiveId = proposal.primitiveId;
												npropPixel128.tile = { dx,dy };
												context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(npropPixel128);
											}
											else {
												int accept[4];
												_mm_storeu_si128((__m128i*)accept, accept128);
												for (int di = 0; di < 4; di++) {
													if (accept[di] == 3 && dx + di % 2 < frameBufferWidth && dy + di / 2 < frameBufferHeight) {
														npropPixel.bbox = proposal.bbox;
														npropPixel.primitiveId = proposal.primitiveId;
														npropPixel.tile = { dx + di % 2, dy + di / 2 };
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
					for (int vx = 0; vx < context->subtileBlocksX * context->subtileBlocksX; vx++) {
						int x = vx % context->subtileBlocksX;
						int y = vx / context->subtileBlocksX;
						int criteriaTR = 0;
						int criteriaTA = 0;

						float wp = (context->subtileBlocksX * context->tileBlocksX);
						float subTileMinX = tileMinX + 1.0f * x / wp;
						float subTileMinY = tileMinY + 1.0f * y / wp;
						float subTileMaxX = tileMinX + 1.0f * (x + 1) / wp;
						float subTileMaxY = tileMinY + 1.0f * (y + 1) / wp;

						int subTilePixelX = (tileIdX*context->subtileBlocksX+ x)* frameBufferWidth / wp;
						int subTilePixelY = (tileIdY*context->subtileBlocksX + y)* frameBufferHeight / wp;
						int subTilePixelX2 = (tileIdX*context->subtileBlocksX + x+1)* frameBufferWidth / wp;
						int subTilePixelY2 = (tileIdY*context->subtileBlocksX + y+1)* frameBufferHeight / wp;

						subTilePixelX2 = std::min(subTilePixelX2 * 1u, frameBufferWidth - 1);
						subTilePixelY2 = std::min(subTilePixelY2 * 1u, frameBufferHeight - 1);

						float3 tileCoords[4];
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
							if (criteriaTRLocal < -EPS) criteriaTR += 1;
							if (criteriaTALocal < EPS) criteriaTA += 1;
						}


						if (criteriaTR != 3) {
							continue;
						}
						if (criteriaTA == 3) {
							TileBinProposal nprop;
							nprop.allAccept = true;
							nprop.level = TileRasterLevel::BLOCK;
							nprop.bbox = proposal.bbox;
							nprop.tile = { x,y };
							nprop.primitiveId = proposal.primitiveId;
							context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(nprop);
						}
						else {
							for (int dx = subTilePixelX; dx <= subTilePixelX2; dx++) {
								for (int dy = subTilePixelY; dy <= subTilePixelY2; dy++) {
									float ndcX = 2.0f * dx / frameBufferWidth - 1.0f;
									float ndcY = 2.0f * dy / frameBufferHeight - 1.0f;
									int accept = 0;
									for (int i = 0; i < 3; i++) {
										float criteria = edgeCoefs[i].x * ndcX + edgeCoefs[i].y * ndcY + edgeCoefs[i].z;
										accept += criteria < EPS2;
									}
									if (accept==3) {
										TileBinProposal nprop;
										nprop.allAccept = true;
										nprop.level = TileRasterLevel::PIXEL;
										nprop.bbox = proposal.bbox;
										nprop.primitiveId = proposal.primitiveId;
										nprop.tile = { dx,dy };
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

		status.store(TileRasterStage::RASTERIZATION_SYNC);
	}


	void TileRasterWorker::fragmentProcessing(){
		auto curTile = 0;
		auto frameBufferWidth = context->frameBuffer->getWidth();
		auto frameBufferHeight = context->frameBuffer->getHeight();
		auto rdTiles = 0;
		interpolatedVaryings.reserve(context->vertexShader->getVaryingCounts());
		interpolatedVaryings.resize(context->vertexShader->getVaryingCounts());
		interpolatedVaryingsAddr.reserve(context->vertexShader->getVaryingCounts());
		interpolatedVaryingsAddr.resize(context->vertexShader->getVaryingCounts());
		for (int i = 0; i < interpolatedVaryingsAddr.size(); i++) {
			interpolatedVaryingsAddr[i] = &interpolatedVaryings[i];
		}
		while ((curTile = renderer->fetchUnresolvedTileFragmentShading()) != -1) {
			for (int i = 0; i < context->numThreads; i++) {
				for (int j = 0; j < context->coverQueue[i][curTile].size(); j++) {
					auto& proposal = context->coverQueue[i][curTile][j];
					if (proposal.level == TileRasterLevel::PIXEL) {
						pixelShading(proposal.primitiveId, proposal.tile.x, proposal.tile.y);
					}
					else if (proposal.level == TileRasterLevel::PIXEL_PACK4X2) {
#ifdef IFRIT_USE_SIMD_128
#ifdef IFRIT_USE_SIMD_256
						pixelShadingSIMD256(proposal.primitiveId, proposal.tile.x, proposal.tile.y);
#else
						for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 3u, frameBufferWidth - 1); dx+=2) {
							for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
								pixelShadingSIMD128(proposal.primitiveId, dx, dy);
							}
						}
#endif
#else
						for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 3u, frameBufferWidth - 1); dx++) {
							for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
								pixelShading(proposal.primitiveId, dx, dy);
							}
						}
#endif
					}
					else if (proposal.level == TileRasterLevel::PIXEL_PACK2X2) {
#ifdef IFRIT_USE_SIMD_128
						pixelShadingSIMD128(proposal.primitiveId, proposal.tile.x, proposal.tile.y);
#else
						for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 1u, frameBufferWidth-1); dx++) {
							for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
								pixelShading(proposal.primitiveId, dx, dy);
							}
						}
#endif
					}
					else if (proposal.level == TileRasterLevel::TILE) {
						auto curTileX = curTile % context->tileBlocksX;
						auto curTileY = curTile / context->tileBlocksX;
						auto curTileX2 = (curTileX + 1) * frameBufferWidth / context->tileBlocksX;
						auto curTileY2 = (curTileY + 1) * frameBufferHeight / context->tileBlocksX;
						curTileX = curTileX * frameBufferWidth/ context->tileBlocksX;
						curTileY = curTileY * frameBufferHeight/ context->tileBlocksX;
						curTileX2 = std::min(curTileX2, frameBufferWidth);
						curTileY2 = std::min(curTileY2, frameBufferHeight);
						for (int dx = curTileX; dx < curTileX2; dx++) {
							for (int dy = curTileY; dy < curTileY2; dy++) {
								pixelShading(proposal.primitiveId, dx, dy);
							}
						}
					}
					else if (proposal.level == TileRasterLevel::BLOCK) {
						auto curTileX = curTile % context->tileBlocksX;
						auto curTileY = curTile / context->tileBlocksX;
						auto subTilePixelX = (curTileX * context->subtileBlocksX + proposal.tile.x) * frameBufferWidth / context->tileBlocksX / context->subtileBlocksX;
						auto subTilePixelY = (curTileY * context->subtileBlocksX + proposal.tile.y) * frameBufferHeight / context->tileBlocksX / context->subtileBlocksX;
						auto subTilePixelX2 = (curTileX * context->subtileBlocksX + proposal.tile.x + 1) * frameBufferWidth / context->tileBlocksX / context->subtileBlocksX;
						auto subTilePixelY2 = (curTileY * context->subtileBlocksX + proposal.tile.y + 1) * frameBufferHeight / context->tileBlocksX / context->subtileBlocksX;
						subTilePixelX2 = std::min(subTilePixelX2, frameBufferWidth);
						subTilePixelY2 = std::min(subTilePixelY2, frameBufferHeight);
						for (int dx = subTilePixelX; dx < subTilePixelX2; dx++) {
							for (int dy = subTilePixelY; dy < subTilePixelY2; dy++) {
								pixelShading(proposal.primitiveId, dx, dy);
							}
						}
					}
				}	
			}
		}
		status.store(TileRasterStage::FRAGMENT_SHADING_SYNC);
	}

	void TileRasterWorker::threadStart(){
		execWorker = std::make_unique<std::thread>(&TileRasterWorker::run, this);
		execWorker->detach();
	}

	void TileRasterWorker::getVertexAttributes(const int id, std::vector<const void*>& out){
		for (int i = 0; i < context->vertexBuffer->getAttributeCount();i++) {
			auto desc = context->vertexBuffer->getAttributeDescriptor(i);
			if (desc.type == TypeDescriptorEnum::FLOAT4) {
				out[i]= (context->vertexBuffer->getValuePtr<float4>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::FLOAT3) {
				out[i] = (context->vertexBuffer->getValuePtr<float3>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::FLOAT2) {
				out[i] = (context->vertexBuffer->getValuePtr<float2>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::FLOAT1) {
				out[i] = (context->vertexBuffer->getValuePtr<float>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::INT1) {
				out[i] = (context->vertexBuffer->getValuePtr<int>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::INT2) {
				out[i] = (context->vertexBuffer->getValuePtr<int2>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::INT3) {
				out[i] = (context->vertexBuffer->getValuePtr<int3>(id, i));
			}
			else if (desc.type == TypeDescriptorEnum::INT4) {
				out[i] = (context->vertexBuffer->getValuePtr<int4>(id, i));
			}
			else {
				ifritError("Unsupported Type");
			}
		}
	}
	void TileRasterWorker::getVaryingsAddr(const int id, std::vector<VaryingStore*>& out) {
		for (int i = 0; i < context->vertexShader->getVaryingCounts(); i++) {
			auto desc = context->vertexShaderResult->getVaryingDescriptor(i);
			out[i] = &context->vertexShaderResult->getVaryingBuffer(i)[id];
		}
	}

	void TileRasterWorker::pixelShading(const int primitiveId, const int dx, const int dy) {
		int idx = primitiveId * context->vertexStride;

		auto& depthAttachment = *context->frameBuffer->getDepthAttachment();
		float referenceDepth = depthAttachment(dx, dy, 0);
		if (context->primitiveMinZ[primitiveId] > referenceDepth) {
			return;
		}

		float4 pos[3];
		pos[0] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx]];
		pos[1] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx + 1]];
		pos[2] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx + 2]];	
		
		float pDx = 2.0f * dx / context->frameBuffer->getWidth() - 1.0f;
		float pDy = 2.0f * dy / context->frameBuffer->getHeight() - 1.0f;
		float4 p = { pDx,pDy,1.0,1.0 };

		float bary[3], area;
		const float w[3] = { pos[0].w,pos[1].w,pos[2].w };
		area = edgeFunction(pos[0], pos[1], pos[2]);
		bary[0] = edgeFunction(pos[1], pos[2], p)/area;
		bary[1] = edgeFunction(pos[2], pos[0], p)/area;
		bary[2] = edgeFunction(pos[0], pos[1], p)/area;
		
		float zCorr = 1.0 / (bary[0] / pos[0].w + bary[1] / pos[1].w + bary[2] / pos[2].w);

		// Interpolate Depth
		float depth[3];
		for (int i = 0; i < 3; i++) {
			depth[i] = pos[i].z / ( pos[i].w);
		}
		float interpolatedDepth = bary[0] * depth[0] + bary[1] * depth[1] + bary[2] * depth[2];
		interpolatedDepth *= zCorr;

		// Depth Test
		if (interpolatedDepth > referenceDepth) {
			return;
		}

		// Interpolate Varyings
		const auto vSize = context->vertexShader->getVaryingCounts();
		bary[0] = bary[0] / w[0] * zCorr;
		bary[1] = bary[1] / w[1] * zCorr;	
		bary[2] = bary[2] / w[2] * zCorr;
		for (int i = 0; i < vSize; i++) {
			 interpolateVaryings(i, (*context->indexBuffer).data() + idx, bary, interpolatedVaryings[i]);
		}
		// Fragment Shader
		context->fragmentShader->execute(interpolatedVaryings, colorOutput);
		context->frameBuffer->getColorAttachment(0)->fillPixelRGBA(dx, dy, colorOutput[0].x, colorOutput[0].y, colorOutput[0].z, colorOutput[0].w);

		// Depth Write
		depthAttachment(dx, dy, 0) = interpolatedDepth;
	}

	void TileRasterWorker::pixelShadingSIMD128(const int primitiveId, const int dx, const int dy) {
#ifndef IFRIT_USE_SIMD_128
		ifritError("SIMD 128 not enabled");
#else
		const auto fbWidth = context->frameBuffer->getWidth();
		const auto fbHeight = context->frameBuffer->getHeight();

		float referenceDepth[4];
		auto& depthAttachment = *context->frameBuffer->getDepthAttachment();
		referenceDepth[0] = depthAttachment(dx, dy, 0);
		referenceDepth[1] = depthAttachment(std::min(dx + 1u,fbWidth-1), dy, 0);
		referenceDepth[2] = depthAttachment(dx, std::min(dy + 1u,fbHeight - 1), 0);
		referenceDepth[3] = depthAttachment(std::min(dx + 1u, fbWidth - 1), std::min(dy + 1u, fbHeight - 1), 0);
		float maxDepth = std::max(std::max(referenceDepth[0], referenceDepth[1]), std::max(referenceDepth[2], referenceDepth[3]));
		float curMinDepthPrimitive = context->primitiveMinZ[primitiveId];
		if (curMinDepthPrimitive > maxDepth) {
			return;
		}

		int idx = primitiveId * context->vertexStride;

		__m128 posX[3], posY[3], posZ[3], posW[3];
		float4 pos[3];
		pos[0] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx]];
		pos[1] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx + 1]];
		pos[2] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx + 2]];
		

		__m128i dx128i = _mm_setr_epi32(dx + 0, dx + 1, dx + 0, dx + 1);
		__m128i dy128i = _mm_setr_epi32(dy + 0, dy + 0, dy + 1, dy + 1);
		float w[3] = { pos[0].w,pos[1].w,pos[2].w };
		for (int i = 0; i < 3; i++) {
			posX[i] = _mm_set1_ps(pos[i].x);
			posY[i] = _mm_set1_ps(pos[i].y);
			posZ[i] = _mm_set1_ps(pos[i].z);
			posW[i] = _mm_set1_ps(pos[i].w);
		}

		__m128 attachmentWidth = _mm_set1_ps(context->frameBuffer->getWidth());
		__m128 attachmentHeight = _mm_set1_ps(context->frameBuffer->getHeight());

		__m128i dpDx = _mm_add_epi32(dx128i, dx128i);
		__m128i dpDy = _mm_add_epi32(dy128i, dy128i);
		__m128 pDx = _mm_sub_ps(_mm_div_ps(_mm_cvtepi32_ps(dpDx), attachmentWidth), _mm_set1_ps(1.0f));
		__m128 pDy = _mm_sub_ps(_mm_div_ps(_mm_cvtepi32_ps(dpDy), attachmentHeight), _mm_set1_ps(1.0f));

		// Edge Function = (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
		__m128 bary[3];
		__m128 area = edgeFunctionSIMD128(posX[0], posY[0], posX[1], posY[1], posX[2], posY[2]);
		bary[0] = edgeFunctionSIMD128(posX[1], posY[1], posX[2], posY[2], pDx, pDy);
		bary[1] = edgeFunctionSIMD128(posX[2], posY[2], posX[0], posY[0], pDx, pDy);
		bary[2] = edgeFunctionSIMD128(posX[0], posY[0], posX[1], posY[1], pDx, pDy);
		bary[0] = _mm_div_ps(bary[0], area);
		bary[1] = _mm_div_ps(bary[1], area);
		bary[2] = _mm_div_ps(bary[2], area);

		__m128 baryDivW[3];
		for (int i = 0; i < 3; i++) {
			baryDivW[i] = _mm_div_ps(bary[i], posW[i]);
		}
		__m128 baryDivWSum = _mm_add_ps(_mm_add_ps(baryDivW[0], baryDivW[1]), baryDivW[2]);
		__m128 zCorr = _mm_div_ps(_mm_set1_ps(1.0f), baryDivWSum);

		__m128 depth[3];
		for (int i = 0; i < 3; i++) {
			depth[i] = _mm_div_ps(posZ[i], posW[i]);
		}
		__m128 interpolatedDepth128 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(bary[0], depth[0]), _mm_mul_ps(bary[1], depth[1])), _mm_mul_ps(bary[2], depth[2]));
		interpolatedDepth128 = _mm_mul_ps(interpolatedDepth128, zCorr);

		float interpolatedDepth[4] = { 0 };
		float bary32[3][4];
		float zCorr32[4];
		_mm_storeu_ps(interpolatedDepth, interpolatedDepth128);
		_mm_storeu_ps(zCorr32, zCorr);
		for (int i = 0; i < 3; i++) {
			_mm_storeu_ps(bary32[i], bary[i]);
		}

		for (int i = 0; i < 4; i++) {
			//Depth Test
			int x = dx + i % 2;
			int y = dy + i / 2;
			if (x >= fbWidth || y >= fbHeight) {
				continue;
			}
			if (interpolatedDepth[i] > depthAttachment(x, y, 0)) {
				continue;
			}
			for (int k = 0; k < context->vertexShader->getVaryingCounts(); k++) {
				float barytmp[3] = { bary32[0][i]/w[0]* zCorr32[i],bary32[1][i] / w[1]* zCorr32[i],bary32[2][i] / w[2]* zCorr32[i] };
				interpolateVaryings(k, (*context->indexBuffer).data() + idx, barytmp,interpolatedVaryings[k]);
			}

			// Fragment Shader
			context->fragmentShader->execute(interpolatedVaryings, colorOutput);
			context->frameBuffer->getColorAttachment(0)->fillPixelRGBA(x, y, colorOutput[0].x, colorOutput[0].y, colorOutput[0].z, colorOutput[0].w);

			// Depth Write
			depthAttachment(x, y, 0) = interpolatedDepth[i];
		}
#endif
	}


	void TileRasterWorker::pixelShadingSIMD256(const int primitiveId, const int dx, const int dy) {
#ifndef IFRIT_USE_SIMD_256
		ifritError("SIMD 256 (AVX2) not enabled");
#else

		int idx = primitiveId * context->vertexStride;

		__m256 posX[3], posY[3], posZ[3], posW[3];
		float4 pos[3];
		pos[0] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx]];
		pos[1] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx + 1]];
		pos[2] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx + 2]];

		__m256i dx256i = _mm256_setr_epi32(dx + 0, dx + 1, dx + 0, dx + 1, dx + 2, dx + 3, dx + 2, dx + 3);
		__m256i dy256i = _mm256_setr_epi32(dy + 0, dy + 0, dy + 1, dy + 1, dy + 0, dy + 0, dy + 1, dy + 1);
		float w[3] = { pos[0].w,pos[1].w,pos[2].w };
		for (int i = 0; i < 3; i++) {
			posX[i] = _mm256_set1_ps(pos[i].x);
			posY[i] = _mm256_set1_ps(pos[i].y);
			posZ[i] = _mm256_set1_ps(pos[i].z);
			posW[i] = _mm256_set1_ps(pos[i].w);
		}

		__m256 attachmentWidth = _mm256_set1_ps(context->frameBuffer->getWidth());
		__m256 attachmentHeight = _mm256_set1_ps(context->frameBuffer->getHeight());

		__m256i dpDx = _mm256_add_epi32(dx256i, dx256i);
		__m256i dpDy = _mm256_add_epi32(dy256i, dy256i);
		__m256 pDx = _mm256_sub_ps(_mm256_div_ps(_mm256_cvtepi32_ps(dpDx), attachmentWidth), _mm256_set1_ps(1.0f));
		__m256 pDy = _mm256_sub_ps(_mm256_div_ps(_mm256_cvtepi32_ps(dpDy), attachmentHeight), _mm256_set1_ps(1.0f));

		// Edge Function = (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
		__m256 bary[3];
		__m256 area = edgeFunctionSIMD256(posX[0], posY[0], posX[1], posY[1], posX[2], posY[2]);
		bary[0] = edgeFunctionSIMD256(posX[1], posY[1], posX[2], posY[2], pDx, pDy);
		bary[1] = edgeFunctionSIMD256(posX[2], posY[2], posX[0], posY[0], pDx, pDy);
		bary[2] = edgeFunctionSIMD256(posX[0], posY[0], posX[1], posY[1], pDx, pDy);
		bary[0] = _mm256_div_ps(bary[0], area);
		bary[1] = _mm256_div_ps(bary[1], area);
		bary[2] = _mm256_div_ps(bary[2], area);

		__m256 baryDivW[3];
		for (int i = 0; i < 3; i++) {
			baryDivW[i] = _mm256_div_ps(bary[i], posW[i]);
		}
		__m256 baryDivWSum = _mm256_add_ps(_mm256_add_ps(baryDivW[0], baryDivW[1]), baryDivW[2]);
		__m256 zCorr = _mm256_div_ps(_mm256_set1_ps(1.0f), baryDivWSum);

		__m256 depth[3];
		for (int i = 0; i < 3; i++) {
			depth[i] = _mm256_div_ps(posZ[i], posW[i]);
		}
		__m256 interpolatedDepth256 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(bary[0], depth[0]), _mm256_mul_ps(bary[1], depth[1])), _mm256_mul_ps(bary[2], depth[2]));
		interpolatedDepth256 = _mm256_mul_ps(interpolatedDepth256, zCorr);

		float interpolatedDepth[8] = { 0 };
		float bary32[3][8];
		float zCorr32[8];
		_mm256_storeu_ps(interpolatedDepth, interpolatedDepth256);
		_mm256_storeu_ps(zCorr32, zCorr);
		for (int i = 0; i < 3; i++) {
			_mm256_storeu_ps(bary32[i], bary[i]);
		}
		const auto fbWidth = context->frameBuffer->getWidth();
		const auto fbHeight = context->frameBuffer->getHeight();
		auto& depthAttachment = *context->frameBuffer->getDepthAttachment();
		for (int i = 0; i < 8; i++) {
			//Depth Test
			int x = dx + (i % 4) % 2 + 2 * (i / 4);
			int y = dy + (i % 4) / 2;
			if(x>= fbWidth || y>= fbHeight){
				continue;
			}
			if (interpolatedDepth[i] > depthAttachment(x, y, 0)) {
				continue;
			}
			for (int k = 0; k < context->vertexShader->getVaryingCounts(); k++) {
				float barytmp[3] = { bary32[0][i] / w[0] * zCorr32[i],bary32[1][i] / w[1] * zCorr32[i],bary32[2][i] / w[2] * zCorr32[i] };
				interpolateVaryings(k, (*context->indexBuffer).data() + idx, barytmp, interpolatedVaryings[k]);
			}

			// Fragment Shader
			context->fragmentShader->execute(interpolatedVaryings, colorOutput);
			context->frameBuffer->getColorAttachment(0)->fillPixelRGBA(x, y, colorOutput[0].x, colorOutput[0].y, colorOutput[0].z, colorOutput[0].w);

			// Depth Write
			depthAttachment(x, y, 0) = interpolatedDepth[i];
		}
#endif
	}

	void TileRasterWorker::interpolateVaryings(int id,const int indices[3], const float barycentric[3], VaryingStore& dest) {
		auto va = context->vertexShaderResult->getVaryingBuffer(id);
		auto varyingDescriptor = context->vertexShaderResult->getVaryingDescriptor(id);

		if (varyingDescriptor.type == TypeDescriptorEnum::FLOAT4) {
			dest.vf4 = { 0,0,0,0 };
			for (int j = 0; j < 3; j++) {
				dest.vf4.x += va[indices[j]].vf4.x * barycentric[j];
				dest.vf4.y += va[indices[j]].vf4.y * barycentric[j];
				dest.vf4.z += va[indices[j]].vf4.z * barycentric[j];
				dest.vf4.w += va[indices[j]].vf4.w * barycentric[j];
			}
		}else if (varyingDescriptor.type == TypeDescriptorEnum::FLOAT3){
			dest.vf3 = { 0,0,0 };
			for (int j = 0; j < 3; j++) {
				dest.vf3.x += va[indices[j]].vf3.x * barycentric[j];
				dest.vf3.y += va[indices[j]].vf3.y * barycentric[j];
				dest.vf3.z += va[indices[j]].vf3.z * barycentric[j];
			}
			
		} else if (varyingDescriptor.type == TypeDescriptorEnum::FLOAT2){
			dest.vf2 = { 0,0 };
			for (int j = 0; j < 3; j++) {
				dest.vf2.x += va[indices[j]].vf2.x * barycentric[j];
				dest.vf2.y += va[indices[j]].vf2.y * barycentric[j];
			}
		}
		else if (varyingDescriptor.type == TypeDescriptorEnum::FLOAT1) {
			dest.vf = 0;
			for (int j = 0; j < 3; j++) {
				dest.vf += va[indices[j]].vf * barycentric[j];
			}
		}
		else {
			ifritError("Unsupported Varying Type");
		}
	}
}