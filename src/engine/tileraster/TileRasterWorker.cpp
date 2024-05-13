#include "engine/tileraster/TileRasterWorker.h"
#include "engine/base/FragmentShader.h"
#include "engine/math/ShaderOps.h"
namespace Ifrit::Engine::TileRaster {
	TileRasterWorker::TileRasterWorker(uint32_t workerId, std::shared_ptr<TileRasterRenderer> renderer, std::shared_ptr<TileRasterContext> context) {
		this->workerId = workerId;
		this->renderer = renderer;
		this->context = context;
	}
	void TileRasterWorker::run() {
		while (true) {
			if (status.load() == TileRasterStage::CREATED || status.load() == TileRasterStage::TERMINATED || activated.load()==false) {
				//context->workerIdleTime[workerId]++;
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
	int TileRasterWorker::triangleHomogeneousClip(const int primitiveId, float4 v1, float4 v2, float4 v3) {
		using Ifrit::Engine::Math::ShaderOps::dot;
		using Ifrit::Engine::Math::ShaderOps::sub;
		using Ifrit::Engine::Math::ShaderOps::add;
		using Ifrit::Engine::Math::ShaderOps::multiply;
		using Ifrit::Engine::Math::ShaderOps::lerp;

		constexpr uint32_t clipIts = 7;
		const float4 clipCriteria[clipIts] = {
			{0,0,0,EPS},
			{1,0,0,0},
			{-1,0,0,0},
			{0,1,0,0},
			{0,-1,0,0},
			{0,0,1,0},
			{0,0,-1,0}
		};
		TileRasterClipVertex ret[2][9];
		int retCur[2] = { 0,3 };

		ret[1][0] = { {1,0,0,0},v1 };
		ret[1][1] = { {0,1,0,0},v2 };
		ret[1][2] = { {0,0,1,0},v3 };

		int clipTimes = 0;
		for (int i = 0; i < clipIts; i++) {
			float4 outNormal = { clipCriteria[i].x,clipCriteria[i].y,clipCriteria[i].z,-1 };
			float4 refPoint = { clipCriteria[i].x,clipCriteria[i].y,clipCriteria[i].z,clipCriteria[i].w };
			retCur[i & 1] = 0;
			const auto psize = retCur[1 - (i & 1)];
			if (psize == 0) {
				return 0;
			}
			int ct = 0;
			auto pc = ret[1 - (i & 1)][0];
			for (int j = 0; j < psize; j++) {
				const auto& pn = ret[1 - (i & 1)][(j + 1) % psize];
				auto npc = dot(pc.pos, outNormal);
				auto npn = dot(pn.pos, outNormal);
			
				if (npc < EPS && npn < EPS) {
					ret[i & 1][retCur[i & 1]++] = pn;
				}
				else if (npc * npn < 0) {
					float4 dir = sub(pn.pos, pc.pos);
					// Solve for t, where W = aX + bY + cZ + d
					// W = a(pc.x + t * dir.x) + b(pc.y + t * dir.y) + c(pc.z + t * dir.z) + d = pc.w + t * dir.w
					// (pc.x*a + pc.y*b + pc.z*c + d - pc.w) + t * (dir.x*a + dir.y*b + dir.z*c - dir.w) = 0
					// t = (pc.w - pc.x*a - pc.y*b - pc.z*c) / (dir.x*a + dir.y*b + dir.z*c - dir.w)
					// a = refPoint.x, b = refPoint.y, c = refPoint.z, d = refPoint.w

					float numo = pc.pos.w - pc.pos.x * refPoint.x - pc.pos.y * refPoint.y - pc.pos.z * refPoint.z;
					float deno = dir.x * refPoint.x + dir.y * refPoint.y + dir.z * refPoint.z - dir.w;
					float t = numo / deno;
					float4 intersection = add(pc.pos, multiply(dir, t));
					float4 barycenter = lerp(pc.barycenter, pn.barycenter, t);

					TileRasterClipVertex newp;
					newp.barycenter = barycenter;
					newp.pos = intersection;
					ret[i & 1][retCur[i & 1]++] = newp;
					if (npn < EPS) {
						ret[i & 1][retCur[i & 1]++] = pn;
					}
				}
				pc = pn;
			}
			if (retCur[i & 1] < 3 && retCur[i & 1] !=0) {
				return 0;
			}
		}
		
		for (int i = 0; i < retCur[clipIts & 1]; i++) {
			ret[clipIts % 2][i].pos.x /= ret[clipIts % 2][i].pos.w;
			ret[clipIts % 2][i].pos.y /= ret[clipIts % 2][i].pos.w;
			ret[clipIts % 2][i].pos.z /= ret[clipIts % 2][i].pos.w;
		}
		auto genTris = 0;
		for (int i = 0; i < retCur[clipIts & 1] - 2; i++) {
			genTris++;
			// (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
			// Equals to c.x*(b.y-a.y) - c.y*(b.x-a.x) + (-a.x*(b.y-a.y)+a.y*(b.x-a.x)
			// Order:23/31/12
			AssembledTriangleProposal atri;
			atri.b1 = ret[clipIts % 2][0].barycenter;
			atri.b2 = ret[clipIts % 2][i + 1].barycenter;
			atri.b3 = ret[clipIts % 2][i + 2].barycenter;
			atri.v1 = ret[clipIts % 2][0].pos;
			atri.v2 = ret[clipIts % 2][i + 1].pos;
			atri.v3 = ret[clipIts % 2][i + 2].pos;
			atri.iw1 = 1 / atri.v1.w;
			atri.iw2 = 1 / atri.v2.w;
			atri.iw3 = 1 / atri.v3.w;

			float ar = 1 / edgeFunction(atri.v1, atri.v2, atri.v3);
			atri.f3 = { (atri.v2.y - atri.v1.y) * ar, -(atri.v2.x - atri.v1.x) * ar,(-atri.v1.x * (atri.v2.y - atri.v1.y) + atri.v1.y * (atri.v2.x - atri.v1.x)) * ar };
			atri.f1 = { (atri.v3.y - atri.v2.y) * ar, -(atri.v3.x - atri.v2.x) * ar,(-atri.v2.x * (atri.v3.y - atri.v2.y) + atri.v2.y * (atri.v3.x - atri.v2.x)) * ar };
			atri.f2 = { (atri.v1.y - atri.v3.y) * ar, -(atri.v1.x - atri.v3.x) * ar,(-atri.v3.x * (atri.v1.y - atri.v3.y) + atri.v3.y * (atri.v1.x - atri.v3.x)) * ar };

			float minx = std::min(atri.v1.x, std::min(atri.v2.x, atri.v3.x));
			float maxx = std::max(atri.v1.x, std::max(atri.v2.x, atri.v3.x));
			float miny = std::min(atri.v1.y, std::min(atri.v2.y, atri.v3.y));
			float maxy = std::max(atri.v1.y, std::max(atri.v2.y, atri.v3.y));
			float minz = std::min(atri.v1.z, std::min(atri.v2.z, atri.v3.z));
			float maxz = std::max(atri.v1.z, std::max(atri.v2.z, atri.v3.z));

			atri.bbox.x = minx;
			atri.bbox.y = miny;
			atri.bbox.w = maxx - minx;
			atri.bbox.h = maxy - miny;

			float3 edgeCoefs[3];
			atri.e1 = { atri.v2.y - atri.v1.y,  atri.v1.x - atri.v2.x,  atri.v2.x * atri.v1.y - atri.v1.x * atri.v2.y };
			atri.e2 = { atri.v3.y - atri.v2.y,  atri.v2.x - atri.v3.x,  atri.v3.x * atri.v2.y - atri.v2.x * atri.v3.y };
			atri.e3 = { atri.v1.y - atri.v3.y,  atri.v3.x - atri.v1.x,  atri.v1.x * atri.v3.y - atri.v3.x * atri.v1.y };

			atri.originalPrimitive = primitiveId;
			context->assembledTriangles[workerId].push_back(atri);
		}
		return genTris;
		
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
	void TileRasterWorker::executeBinner(const int primitiveId, const AssembledTriangleProposal& atp) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		const float tileSize = 1.0 / context->tileBlocksX;
		const auto& bbox = atp.bbox;
		float minx = atp.bbox.x;
		float miny = bbox.y * 0.5 + 0.5;
		float maxx = (bbox.x + bbox.w) * 0.5 + 0.5;
		float maxy = (bbox.y + bbox.h) * 0.5 + 0.5;

		int tileMinx = std::max(0, (int)(minx / tileSize));
		int tileMiny = std::max(0, (int)(miny / tileSize));
		int tileMaxx = std::min(context->tileBlocksX - 1, (int)(maxx / tileSize));
		int tileMaxy = std::min(context->tileBlocksX - 1, (int)(maxy / tileSize));

		float4 v1 = atp.v1;
		float4 v2 = atp.v2;
		float4 v3 = atp.v3;

		float3 edgeCoefs[3];
		edgeCoefs[0] = { v2.y - v1.y, v1.x - v2.x, v2.x * v1.y - v1.x * v2.y };
		edgeCoefs[1] = { v3.y - v2.y, v2.x - v3.x, v3.x * v2.y - v2.x * v3.y };
		edgeCoefs[2] = { v1.y - v3.y, v3.x - v1.x, v1.x * v3.y - v3.x * v1.y };

		float3 tileCoords[4];

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		const auto frameBufferWidth = context->frameBuffer->getWidth();
		const auto frameBufferHeight = context->frameBuffer->getHeight();
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
					if (criteriaTRLocal > 0) break;
					if (criteriaTALocal < 0) criteriaTA += 1;
				}
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
		}
		status.store(TileRasterStage::VERTEX_SHADING_SYNC);
	}

	void TileRasterWorker::geometryProcessing() {
		auto posBuffer = context->vertexShaderResult->getPositionBuffer();
		generatedTriangle.clear();
		auto totalGenTriangles = 0;
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
			
			auto fw = totalGenTriangles + triangleHomogeneousClip(prim, v1, v2, v3);
			for(int i = totalGenTriangles; i < fw; i++) {
				const auto& atri = context->assembledTriangles[workerId][i];
				rect2Df bbox;
				if (!triangleFrustumClip(atri.v1, atri.v2, atri.v3, bbox)) {
					continue;
				}
				if (!triangleCulling(atri.v1, atri.v2, atri.v3)) {
					continue;
				}
				executeBinner(i, atri);
			}
			totalGenTriangles = fw;
			context->primitiveMinZ[prim] = 0;
		}
		status.store(TileRasterStage::GEOMETRY_PROCESSING_SYNC);
	}

	void TileRasterWorker::rasterization(){
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		auto curTile = 0;
		const auto frameBufferWidth = context->frameBuffer->getWidth();
		const auto frameBufferHeight = context->frameBuffer->getHeight();
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
					const auto& proposal = context->rasterizerQueue[T][curTile][j];
					const auto& ptRef = context->assembledTriangles[proposal.clippedTriangle.workerId][proposal.clippedTriangle.primId];
					float4 v1 = ptRef.v1;
					float4 v2 = ptRef.v2;
					float4 v3 = ptRef.v3;

					if (context->frontface == TileRasterFrontFace::COUNTER_CLOCKWISE) {
						std::swap(v1, v3);
					}
					//auto edgeCoefs = context->primitiveEdgeCoefs[proposal.primitiveId].coef;
					float3 edgeCoefs[3];
					edgeCoefs[0] = { v2.y - v1.y, v1.x - v2.x, v2.x * v1.y - v1.x * v2.y };
					edgeCoefs[1] = { v3.y - v2.y, v2.x - v3.x, v3.x * v2.y - v2.x * v3.y };
					edgeCoefs[2] = { v1.y - v3.y, v3.x - v1.x, v1.x * v3.y - v3.x * v1.y };


					int chosenCoordTR[3];
					int chosenCoordTA[3];
					getAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

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
					TileBinProposal npropBlock;
					npropBlock.level = TileRasterLevel::BLOCK;
					npropBlock.clippedTriangle = proposal.clippedTriangle;

					TileBinProposal npropPixel;
					npropPixel.level = TileRasterLevel::PIXEL;
					npropPixel.clippedTriangle = proposal.clippedTriangle;

					TileBinProposal  npropPixel128;
					npropPixel128.level = TileRasterLevel::PIXEL_PACK2X2;
					npropPixel128.clippedTriangle = proposal.clippedTriangle;

					TileBinProposal  npropPixel256;
					npropPixel256.level = TileRasterLevel::PIXEL_PACK4X2;
					npropPixel256.clippedTriangle = proposal.clippedTriangle;

					const auto tileId = getTileID(tileIdX, tileIdY);

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
								auto dwX = x + (i & 1);
								auto dwY = y + (i >> 1);
								if (criteriaTR[i] != 3 || (dwX > rightBlock || dwY > bottomBlock)) {
									continue;
								}
								
								if (criteriaTA[i] == 3) {
									npropBlock.tile = { x + (i & 1), y + (i >> 1) };
									context->coverQueue[workerId][tileId].push_back(npropBlock);
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
												npropPixel256.tile = { dx,dy };
												context->coverQueue[workerId][tileId].push_back(npropPixel256);
											}
											else {
												// Pack By 2
												__m128i accept128[2];
												_mm256_storeu_si256((__m256i*)accept128, accept256);
												for (int di = 0; di < 2; di++) {
													const auto pv = dx + ((di & 1) << 1);
													if (pv <= subTileMaxX && dy<= subTileMaxY &&
														_mm_movemask_epi8(_mm_cmpeq_epi32(accept128[di], _mm_set1_epi32(3))) == 0xFFFF) {
														npropPixel128.tile = { pv, dy  };
														context->coverQueue[workerId][tileId].push_back(npropPixel128);
													}
													else {
														int accept[4];
														_mm_storeu_si128((__m128i*)accept, accept128[di]);
														
														for (int ddi = 0; ddi < 4; ddi++) {
															const auto pvx = pv + (ddi & 1);
															const auto pvy = dy + (ddi >> 1);
															if (pvx <= subTileMaxX && pvy <= subTileMaxY && accept[ddi] == 3) {
																npropPixel.tile = { pvx,pvy };
																context->coverQueue[workerId][tileId].push_back(npropPixel);
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
												npropPixel128.clippedTriangle = proposal.clippedTriangle;
												context->coverQueue[workerId][getTileID(tileIdX, tileIdY)].push_back(npropPixel128);
											}
											else {
												int accept[4];
												_mm_storeu_si128((__m128i*)accept, accept128);
												for (int di = 0; di < 4; di++) {
													if (accept[di] == 3 && dx + di % 2 < frameBufferWidth && dy + di / 2 < frameBufferHeight) {
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
							nprop.clippedTriangle = proposal.clippedTriangle;
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
					const auto& triProposal = context->assembledTriangles[proposal.clippedTriangle.workerId][proposal.clippedTriangle.primId];
					if (proposal.level == TileRasterLevel::PIXEL) {
						pixelShading(triProposal, proposal.tile.x, proposal.tile.y);
					}
					else if (proposal.level == TileRasterLevel::PIXEL_PACK4X2) {
#ifdef IFRIT_USE_SIMD_128
#ifdef IFRIT_USE_SIMD_256
						pixelShadingSIMD256(triProposal, proposal.tile.x, proposal.tile.y);
#else
						for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 3u, frameBufferWidth - 1); dx+=2) {
							for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
								pixelShadingSIMD128(triProposal, dx, dy);
							}
						}
#endif
#else
						for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 3u, frameBufferWidth - 1); dx++) {
							for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
								pixelShading(triProposal, dx, dy);
							}
						}
#endif
					}
					else if (proposal.level == TileRasterLevel::PIXEL_PACK2X2) {
#ifdef IFRIT_USE_SIMD_128
						pixelShadingSIMD128(triProposal, proposal.tile.x, proposal.tile.y);
#else
						for (int dx = proposal.tile.x; dx <= std::min(proposal.tile.x + 1u, frameBufferWidth-1); dx++) {
							for (int dy = proposal.tile.y; dy <= std::min(proposal.tile.y + 1u, frameBufferHeight - 1); dy++) {
								pixelShading(triProposal, dx, dy);
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
								pixelShading(triProposal, dx, dy);
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
#ifdef IFRIT_USE_SIMD_128
						for (int dx = subTilePixelX; dx < subTilePixelX2; dx+=2) {
							for (int dy = subTilePixelY; dy < subTilePixelY2; dy+=2) {
								pixelShadingSIMD128(triProposal, dx, dy);
							}
						}
#else
						for (int dx = subTilePixelX; dx < subTilePixelX2; dx++) {
							for (int dy = subTilePixelY; dy < subTilePixelY2; dy++) {
								pixelShading(triProposal, dx, dy);
							}
						}
#endif
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

	void TileRasterWorker::pixelShading(const AssembledTriangleProposal& atp, const int dx, const int dy) {
		
		auto& depthAttachment = *context->frameBuffer->getDepthAttachment();
		float referenceDepth = depthAttachment(dx, dy, 0);

		auto originalPrimitive = atp.originalPrimitive;
		int idx = originalPrimitive * context->vertexStride;

		float4 pos[3];
		pos[0] = atp.v1;
		pos[1] = atp.v2;
		pos[2] = atp.v3;	
		
		float pDx = 2.0f * dx / context->frameBuffer->getWidth() - 1.0f;
		float pDy = 2.0f * dy / context->frameBuffer->getHeight() - 1.0f;
		float4 p = { pDx,pDy,1.0,1.0 };

		float bary[3];
		bary[0] = atp.f1.x * p.x + atp.f1.y * p.y + atp.f1.z;
		bary[1] = atp.f2.x * p.x + atp.f2.y * p.y + atp.f2.z;
		bary[2] = atp.f3.x * p.x + atp.f3.y * p.y + atp.f3.z;
		
		float zCorr = 1.0 / (bary[0] * atp.iw1 + bary[1] * atp.iw2 + bary[2] * atp.iw3);

		// Interpolate Depth
		float depth[3];
		depth[0] = pos[0].z * atp.iw1;
		depth[1] = pos[1].z * atp.iw2;
		depth[2] = pos[2].z * atp.iw3;

		float interpolatedDepth = bary[0] * depth[0] + bary[1] * depth[1] + bary[2] * depth[2];
		interpolatedDepth *= zCorr;

		// Depth Test
		if (interpolatedDepth > referenceDepth) {
			return;
		}

		// Interpolate Varyings
		const auto vSize = context->vertexShader->getVaryingCounts();
		bary[0] = bary[0] * atp.iw1 * zCorr;
		bary[1] = bary[1] * atp.iw2 * zCorr;
		bary[2] = bary[2] * atp.iw3 * zCorr;

		float desiredBary[3];
		desiredBary[0] = bary[0] * atp.b1.x + bary[1] * atp.b2.x + bary[2] * atp.b3.x;
		desiredBary[1] = bary[0] * atp.b1.y + bary[1] * atp.b2.y + bary[2] * atp.b3.y;
		desiredBary[2] = bary[0] * atp.b1.z + bary[1] * atp.b2.z + bary[2] * atp.b3.z;

		const int* const indicesAddr = (*context->indexBuffer).data() + idx;
		for (int i = 0; i < vSize; i++) {
			 interpolateVaryings(i,indicesAddr, desiredBary, interpolatedVaryings[i]);
		}
		// Fragment Shader
		context->fragmentShader->execute(interpolatedVaryings, colorOutput);
		context->frameBuffer->getColorAttachment(0)->fillPixelRGBA(dx, dy, colorOutput[0].x, colorOutput[0].y, colorOutput[0].z, colorOutput[0].w);

		// Depth Write
		depthAttachment(dx, dy, 0) = interpolatedDepth;
	}

	void TileRasterWorker::pixelShadingSIMD128(const AssembledTriangleProposal& atp, const int dx, const int dy) {
#ifndef IFRIT_USE_SIMD_128
		ifritError("SIMD 128 not enabled");
#else
		const auto fbWidth = context->frameBuffer->getWidth();
		const auto fbHeight = context->frameBuffer->getHeight();

		float referenceDepth[4];
		auto& depthAttachment = *context->frameBuffer->getDepthAttachment();

		int idx = atp.originalPrimitive * context->vertexStride;

		__m128 posX[3], posY[3], posZ[3], posW[3], atriW[3];
		float4 pos[3];
		pos[0] = atp.v1;
		pos[1] = atp.v2;
		pos[2] = atp.v3;

		__m128i dx128i = _mm_setr_epi32(dx + 0, dx + 1, dx + 0, dx + 1);
		__m128i dy128i = _mm_setr_epi32(dy + 0, dy + 0, dy + 1, dy + 1);
		float w[3] = { pos[0].w,pos[1].w,pos[2].w };
		for (int i = 0; i < 3; i++) {
			posX[i] = _mm_set1_ps(pos[i].x);
			posY[i] = _mm_set1_ps(pos[i].y);
			posZ[i] = _mm_set1_ps(pos[i].z);
			posW[i] = _mm_set1_ps(pos[i].w);
		}
		atriW[0] = _mm_set1_ps(atp.iw1);
		atriW[1] = _mm_set1_ps(atp.iw2);
		atriW[2] = _mm_set1_ps(atp.iw3);

		__m128 attachmentWidth = _mm_set1_ps(context->frameBuffer->getWidth());
		__m128 attachmentHeight = _mm_set1_ps(context->frameBuffer->getHeight());

		__m128i dpDx = _mm_add_epi32(dx128i, dx128i);
		__m128i dpDy = _mm_add_epi32(dy128i, dy128i);
		__m128 pDx = _mm_sub_ps(_mm_div_ps(_mm_cvtepi32_ps(dpDx), attachmentWidth), _mm_set1_ps(1.0f));
		__m128 pDy = _mm_sub_ps(_mm_div_ps(_mm_cvtepi32_ps(dpDy), attachmentHeight), _mm_set1_ps(1.0f));

		__m128 bary[3];
		bary[0] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(atp.f1.x), pDx), _mm_mul_ps(_mm_set1_ps(atp.f1.y), pDy)), _mm_set1_ps(atp.f1.z));
		bary[1] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(atp.f2.x), pDx), _mm_mul_ps(_mm_set1_ps(atp.f2.y), pDy)), _mm_set1_ps(atp.f2.z));
		bary[2] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(atp.f3.x), pDx), _mm_mul_ps(_mm_set1_ps(atp.f3.y), pDy)), _mm_set1_ps(atp.f3.z));

		__m128 baryDivW[3];
		baryDivW[0] = _mm_mul_ps(bary[0], atriW[0]);
		baryDivW[1] = _mm_mul_ps(bary[1], atriW[1]);
		baryDivW[2] = _mm_mul_ps(bary[2], atriW[2]);

		__m128 baryDivWSum = _mm_add_ps(_mm_add_ps(baryDivW[0], baryDivW[1]), baryDivW[2]);
		__m128 zCorr = _mm_div_ps(_mm_set1_ps(1.0f), baryDivWSum);

		__m128 depth[3];
		depth[0] = _mm_mul_ps(posZ[0], atriW[0]);
		depth[1] = _mm_mul_ps(posZ[1], atriW[1]);
		depth[2] = _mm_mul_ps(posZ[2], atriW[2]);

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

		const int* const indicesAddr = (*context->indexBuffer).data() + idx;
		for (int i = 0; i < 4; i++) {
			//Depth Test
			int x = dx + (i & 1);
			int y = dy + (i >> 1);
			if (x >= fbWidth || y >= fbHeight || interpolatedDepth[i] > depthAttachment(x, y, 0)) {
				continue;
			}
			for (int k = 0; k < context->vertexShader->getVaryingCounts(); k++) {
				float barytmp[3] = { 
					bary32[0][i] * atp.iw1 * zCorr32[i],
					bary32[1][i] * atp.iw2 * zCorr32[i],
					bary32[2][i] * atp.iw3 * zCorr32[i]
				};
				float desiredBary[3];
				desiredBary[0] = barytmp[0] * atp.b1.x + barytmp[1] * atp.b2.x + barytmp[2] * atp.b3.x;
				desiredBary[1] = barytmp[0] * atp.b1.y + barytmp[1] * atp.b2.y + barytmp[2] * atp.b3.y;
				desiredBary[2] = barytmp[0] * atp.b1.z + barytmp[1] * atp.b2.z + barytmp[2] * atp.b3.z;
				interpolateVaryings(k, indicesAddr, desiredBary, interpolatedVaryings[k]);
			}

			// Fragment Shader
			context->fragmentShader->execute(interpolatedVaryings, colorOutput);
			context->frameBuffer->getColorAttachment(0)->fillPixelRGBA(x, y, colorOutput[0].x, colorOutput[0].y, colorOutput[0].z, colorOutput[0].w);

			// Depth Write
			depthAttachment(x, y, 0) = interpolatedDepth[i];
		}
#endif
	}


	void TileRasterWorker::pixelShadingSIMD256(const AssembledTriangleProposal& atp, const int dx, const int dy) {
#ifndef IFRIT_USE_SIMD_256
		ifritError("SIMD 256 (AVX2) not enabled");
#else

		int idx = atp.originalPrimitive * context->vertexStride;

		__m256 posX[3], posY[3], posZ[3], posW[3];
		float4 pos[3];
		pos[0] = atp.v1;
		pos[1] = atp.v2;
		pos[2] = atp.v3;

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
				float barytmp[3] = { bary32[0][i] * atp.iw1 * zCorr32[i],bary32[1][i] * atp.iw2 * zCorr32[i],bary32[2][i] * atp.iw3 * zCorr32[i] };
				float desiredBary[3];
				desiredBary[0] = barytmp[0] * atp.b1.x + barytmp[1] * atp.b2.x + barytmp[2] * atp.b3.x;
				desiredBary[1] = barytmp[0] * atp.b1.y + barytmp[1] * atp.b2.y + barytmp[2] * atp.b3.y;
				desiredBary[2] = barytmp[0] * atp.b1.z + barytmp[1] * atp.b2.z + barytmp[2] * atp.b3.z;
				interpolateVaryings(k, (*context->indexBuffer).data() + idx, desiredBary, interpolatedVaryings[k]);
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