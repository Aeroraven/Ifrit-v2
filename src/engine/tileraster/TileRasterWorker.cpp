#include "engine/tileraster/TileRasterWorker.h"

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

		float3 edgeCoefs[3];
		edgeCoefs[0].x = v2.y - v1.y;
		edgeCoefs[0].y = v1.x - v2.x;
		edgeCoefs[0].z = v2.x * v1.y - v1.x * v2.y;
		edgeCoefs[1].x = v3.y - v2.y;
		edgeCoefs[1].y = v2.x - v3.x;
		edgeCoefs[1].z = v3.x * v2.y - v2.x * v3.y;
		edgeCoefs[2].x = v1.y - v3.y;
		edgeCoefs[2].y = v3.x - v1.x;
		edgeCoefs[2].z = v1.x * v3.y - v3.x * v1.y;

		for (int i = 0; i < 3; i++) {
			auto norm = sqrt(edgeCoefs[i].x * edgeCoefs[i].x + edgeCoefs[i].y * edgeCoefs[i].y);
			edgeCoefs[i].x /= norm;
			edgeCoefs[i].y /= norm;
			edgeCoefs[i].z /= norm;
		}

		float3 tileCoords[4];

		int chosenCoordTR[3];
		int chosenCoordTA[3];


		for (int i = 0; i < 3; i++) {
			bool normalRight = edgeCoefs[i].x > 0;
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

		for (int y = tileMiny; y <= tileMaxy; y++) {
			for (int x = tileMinx; x <= tileMaxx; x++) {
				int criteriaTR = 0;
				int criteriaTA = 0;

				auto frameBufferWidth = context->frameBuffer->getColorAttachment(0)->getWidth();
				auto frameBufferHeight = context->frameBuffer->getColorAttachment(0)->getHeight();
				
				auto curTileX = x * frameBufferWidth / context->tileBlocksX;
				auto curTileY = y * frameBufferHeight / context->tileBlocksX;
				auto curTileX2 = (x + 1) * frameBufferWidth / context->tileBlocksX;
				auto curTileY2 = (y + 1) * frameBufferHeight / context->tileBlocksX;

				tileCoords[VLB] = { x * tileSize, y * tileSize, 1.0 };
				tileCoords[VLT] = { x * tileSize, (y + 1) * tileSize, 1.0 };
				tileCoords[VRT] = { (x + 1) * tileSize, (y + 1) * tileSize, 1.0 };
				tileCoords[VRB] = { (x + 1) * tileSize, y * tileSize, 1.0 };

				for (int i = 0; i < 4; i++) {
					tileCoords[i].x = tileCoords[i].x * 2 - 1;
					tileCoords[i].y = tileCoords[i].y * 2 - 1;
				}

				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoords[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTR[i]].y + edgeCoefs[i].z;
					float criteriaTALocal = edgeCoefs[i].x * tileCoords[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTA[i]].y + edgeCoefs[i].z;
					if (criteriaTR > 0) criteriaTR += 1;
					if (criteriaTA > 0) criteriaTA += 1;
					if (i == 1) {
						ifritLog1(criteriaTALocal, tileCoords[i].y);
					}
				}
				context->frameBuffer->getColorAttachment(0)->fillArea(curTileX,curTileY, curTileX2-curTileX, curTileY2-curTileY, 100.0);
				if (criteriaTR != 3)continue;
				if (criteriaTA == 3) {
					TileBinProposal proposal;
					proposal.allAccept = true;
					proposal.bbox = bbox;
					proposal.primitiveId = primitiveId;
					context->rasterizerQueue[workerId].push_back(proposal);

					context->frameBuffer->getColorAttachment(0)->fillArea(curTileX, curTileY, curTileX2 - curTileX, curTileY2 - curTileY, 200.0);
				}
				else {
					TileBinProposal proposal;
					proposal.allAccept = false;
					proposal.bbox = bbox;
					proposal.primitiveId = primitiveId;
					context->frameBuffer->getColorAttachment(0)->fillArea(curTileX, curTileY, curTileX2 - curTileX, curTileY2 - curTileY, 250.0);
				}
			}
		}
		
	}
	void TileRasterWorker::vertexProcessing() {
		status.store(TileRasterStage::VERTEX_SHADING);
		for (int j = workerId; j < context->indexBuffer->size(); j += context->numThreads) {
			auto id = (*context->indexBuffer)[j];
			context->vertexShader->execute(id);
			auto pos = context->vertexShaderResult->getPositionBuffer()[id];
			auto w = pos.w;
			pos.x /= w;
			pos.y /= w;
			pos.z /= w;
		}
		status.store(TileRasterStage::VERTEX_SHADING_SYNC);
	}

	void TileRasterWorker::geometryProcessing() {
		auto posBuffer = context->vertexShaderResult->getPositionBuffer();
		for (int j = workerId * context->vertexStride; j < context->indexBuffer->size(); j += context->numThreads * context->vertexStride) {
			float4 v1 = posBuffer[(*context->indexBuffer)[j]];
			float4 v2 = posBuffer[(*context->indexBuffer)[j + 1]];
			float4 v3 = posBuffer[(*context->indexBuffer)[j + 2]];
			rect2Df bbox;
			if (!triangleFrustumClip(v1, v2, v3, bbox)) {
				continue;
			}
			if (!triangleCulling(v1, v2, v3)) {
				continue;
			}
			executeBinner(j / context->vertexStride, v1, v2, v3, bbox);
		}
		status.store(TileRasterStage::GEOMETRY_PROCESSING_SYNC);
	}

	void TileRasterWorker::rasterization(){
		status.store(TileRasterStage::RASTERIZATION_SYNC);
	}

	void TileRasterWorker::fragmentProcessing(){
		status.store(TileRasterStage::FRAGMENT_SHADING_SYNC);
	}

	void TileRasterWorker::threadStart(){
		execWorker = std::make_unique<std::thread>(&TileRasterWorker::run, this);
		execWorker->detach();
	}



}