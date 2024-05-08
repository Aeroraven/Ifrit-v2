#include "engine/tileraster/TileRasterRenderer.h"

namespace Ifrit::Engine::TileRaster {
	void TileRasterRenderer::bindFrameBuffer(FrameBuffer& frameBuffer) {
		this->frameBuffer = &frameBuffer;
	}
	void TileRasterRenderer::bindVertexBuffer(const VertexBuffer& vertexBuffer) {
		this->vertexBuffer = &vertexBuffer;
	}
	void TileRasterRenderer::bindIndexBuffer(const std::vector<int>& indexBuffer) {
		this->indexBuffer = &indexBuffer;
	}
	void TileRasterRenderer::bindVertexShader(VertexShader& vertexShader) {
		this->vertexShader = &vertexShader;
	}
	bool TileRasterRenderer::triangleFrustumClip(float4 v1, float4 v2, float4 v3, rect2Df& bbox) {
		bool inside = false;
		if(v1.x>0.0 && v1.x<1.0 && v1.y>0.0 && v1.y<1.0 && v1.z>0.0 && v1.z<1.0) inside = true;
		if(v2.x>0.0 && v2.x<1.0 && v2.y>0.0 && v2.y<1.0 && v2.z>0.0 && v2.z<1.0) inside = true;
		if(v3.x>0.0 && v3.x<1.0 && v3.y>0.0 && v3.y<1.0 && v3.z>0.0 && v3.z<1.0) inside = true;
		if(!inside) return false;
		float minx = std::min(v1.x, std::min(v2.x, v3.x));
		float miny = std::min(v1.y, std::min(v2.y, v3.y));
		float maxx = std::max(v1.x, std::max(v2.x, v3.x));
		float maxy = std::max(v1.y, std::max(v2.y, v3.y));
		bbox.x = minx;
		bbox.y = miny;
		bbox.w = maxx - minx;
		bbox.h = maxy - miny;
		return true;
	}
	bool TileRasterRenderer::triangleCulling(float4 v1, float4 v2, float4 v3) {
		float d1 = (v1.x * v2.y);
		float d2 = (v2.x * v3.y);
		float d3 = (v3.x * v1.y);
		float n1 = (v3.x * v2.y);
		float n2 = (v1.x * v3.y);
		float n3 = (v2.x * v1.y);
		float d = d1 + d2 + d3 - n1 - n2 - n3;
		if(d<0.0) return true;
		return false;
	}
	void TileRasterRenderer::executeBinner(const int threadId, const int primitiveId, float4 v1, float4 v2, float4 v3, rect2Df bbox) {
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		const float tileSize = 1.0/tileBlocksX;
		float minx = bbox.x * 0.5 + 0.5;
		float miny = bbox.y * 0.5 + 0.5;
		float maxx = (bbox.x + bbox.w) * 0.5 + 0.5;
		float maxy = (bbox.y + bbox.h) * 0.5 + 0.5;
		
		int tileMinx = std::max(0, (int)(minx / tileSize));
		int tileMiny = std::max(0, (int)(miny / tileSize));
		int tileMaxx = std::min(tileBlocksX - 1, (int)(maxx / tileSize));
		int tileMaxy = std::min(tileBlocksX - 1, (int)(maxy / tileSize));

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

				tileCoords[VLB]= { x * tileSize, y * tileSize, 1.0 };
				tileCoords[VLT] = { x * tileSize, (y + 1) * tileSize, 1.0 };
				tileCoords[VRT] = { (x + 1) * tileSize, (y + 1) * tileSize, 1.0 };
				tileCoords[VRB] = { (x + 1) * tileSize, y * tileSize, 1.0 };

				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * tileCoords[chosenCoordTR[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTR[i]].y + edgeCoefs[i].z;
					float criteriaTALocal = edgeCoefs[i].x * tileCoords[chosenCoordTA[i]].x + edgeCoefs[i].y * tileCoords[chosenCoordTA[i]].y + edgeCoefs[i].z;
					if (criteriaTR > 0) criteriaTR += 1;
					if (criteriaTA > 0) criteriaTA += 1;
				}
				if (criteriaTR != 3)return;
				if (criteriaTA == 3) {
					TileBinProposal proposal;
					proposal.allAccept = true;
					proposal.bbox = bbox;
					proposal.primitiveId = primitiveId;
					rasterizerQueue[threadId].push_back(proposal);
				}
				else {
					TileBinProposal proposal;
					proposal.allAccept = false;
					proposal.bbox = bbox;
					proposal.primitiveId = primitiveId;
				}
			}
		}

	}
	void TileRasterRenderer::render() {
		
		// Run Vertex Shader
		std::vector<std::thread> threads;
		for (int i = 0; i < numThreads; i++) {
			threads.push_back(std::thread([this, i]() {
				for (int j = i; j < indexBuffer->size(); j += numThreads) {
					auto id = (*indexBuffer)[j];
					vertexShader->execute(id);
					auto pos = vertexShaderResult->getPositionBuffer()[id];
					auto w = pos.w;
					pos.x /= w;
					pos.y /= w;
					pos.z /= w;
				}
			}));
		}
		for (auto& thread : threads) {
			thread.join();
		}

		// Geometry Processing
		threads.clear();
		auto posBuffer = vertexShaderResult->getPositionBuffer();
		for (int i = 0; i < numThreads; i++) {
			threads.push_back(std::thread([this, i, posBuffer]() {
				for (int j = i* vertexStride; j < indexBuffer->size(); j += numThreads * vertexStride) {
					float4 v1 = posBuffer[(*indexBuffer)[j]];
					float4 v2 = posBuffer[(*indexBuffer)[j + 1]];
					float4 v3 = posBuffer[(*indexBuffer)[j + 2]];
					rect2Df bbox;
					if (!triangleFrustumClip(v1, v2, v3, bbox)) {
						continue;
					}
					if (!triangleCulling(v1, v2, v3)) {
						continue;
					}
					executeBinner(i, j / vertexStride, v1, v2, v3, bbox);
				}
			}));
		}
		for (auto& thread : threads) {
			thread.join();
		}

	}

}