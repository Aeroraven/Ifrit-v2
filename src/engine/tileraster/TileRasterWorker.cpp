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
			auto norm = std::max(abs(edgeCoefs[i].x), abs(edgeCoefs[i].y));
			edgeCoefs[i].x /= norm;
			edgeCoefs[i].y /= norm;
			edgeCoefs[i].z /= norm;
		}

		float3 tileCoords[4];

		int chosenCoordTR[3];
		int chosenCoordTA[3];
		auto frameBufferWidth = context->frameBuffer->getColorAttachment(0)->getWidth();
		auto frameBufferHeight = context->frameBuffer->getColorAttachment(0)->getHeight();
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
				//context->frameBuffer->getColorAttachment(0)->fillArea(curTileX, curTileY, curTileX2- curTileX, curTileY2- curTileY,150);
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
		status.store(TileRasterStage::VERTEX_SHADING);
		for (int j = workerId; j < context->vertexBuffer->getVertexCount(); j += context->numThreads) {
			context->vertexShader->execute(j);
			auto pos = context->vertexShaderResult->getPositionBuffer()[j];
			auto w = pos.w;
		}
		status.store(TileRasterStage::VERTEX_SHADING_SYNC);
	}

	void TileRasterWorker::geometryProcessing() {
		auto posBuffer = context->vertexShaderResult->getPositionBuffer();
		for (int j = workerId * context->vertexStride; j < context->indexBuffer->size(); j += context->numThreads * context->vertexStride) {
			int id0 = (*context->indexBuffer)[j];
			int id1 = (*context->indexBuffer)[j + 1];
			int id2 = (*context->indexBuffer)[j + 2];
			float4 v1 = posBuffer[id0];
			float4 v2 = posBuffer[id1];
			float4 v3 = posBuffer[id2];

			v1.x /= v1.w;
			v1.y /= v1.w;
			v1.z /= v1.w;
			v2.x /= v2.w;
			v2.y /= v2.w;
			v2.z /= v2.w;
			v3.x /= v3.w;
			v3.y /= v3.w;
			v3.z /= v3.w;

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
		constexpr const int VLB = 0, VLT = 1, VRT = 2, VRB = 3;

		auto curTile = 0;
		auto frameBufferWidth = context->frameBuffer->getColorAttachment(0)->getWidth();
		auto frameBufferHeight = context->frameBuffer->getColorAttachment(0)->getHeight();
		auto rdTiles = 0;
		while ((curTile = renderer->fetchUnresolvedTileRaster()) != -1) {
			rdTiles++;
			int tileIdX = curTile % context->tileBlocksX;
			int tileIdY = curTile / context->tileBlocksX;

			float tileMinX = 1.0f * tileIdX / context->tileBlocksX;
			float tileMinY = 1.0f * tileIdY / context->tileBlocksX;
			float tileMaxX = 1.0f * (tileIdX + 1) / context->tileBlocksX;
			float tileMaxY = 1.0f * (tileIdY + 1) / context->tileBlocksX;


			for (int i = 0; i < context->numThreads; i++) {
				for (int j = 0; j < context->rasterizerQueue[i][curTile].size(); j++) {
					auto& proposal = context->rasterizerQueue[i][curTile][j];
					int idx0 = proposal.primitiveId * context->vertexStride;
					int idx1 = idx0 + 1;
					int idx2 = idx0 + 2;
					float4 v1 = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx0]];
					float4 v2 = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx1]];
					float4 v3 = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx2]];

					v1.x /= v1.w;
					v1.y /= v1.w;
					v1.z /= v1.w;
					v2.x /= v2.w;
					v2.y /= v2.w;
					v2.z /= v2.w;
					v3.x /= v3.w;
					v3.y /= v3.w;
					v3.z /= v3.w;

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

					for (int k = 0; k < 3; k++) {
						auto norm = std::max(abs(edgeCoefs[k].x), abs(edgeCoefs[k].y));
						edgeCoefs[k].x /= norm;
						edgeCoefs[k].y /= norm;
						edgeCoefs[k].z /= norm;
					}

					int chosenCoordTR[3];
					int chosenCoordTA[3];
					getAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

					for (int x = 0; x < context->subtileBlocksX; x++) {
						for (int y = 0; y < context->subtileBlocksX; y++) {
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
					}
				}
			}
		}
		status.store(TileRasterStage::RASTERIZATION_SYNC);
	}

	void TileRasterWorker::fragmentProcessing(){
		auto curTile = 0;
		auto frameBufferWidth = context->frameBuffer->getColorAttachment(0)->getWidth();
		auto frameBufferHeight = context->frameBuffer->getColorAttachment(0)->getHeight();
		auto rdTiles = 0;
		interpolatedVaryings.resize(context->vertexShader->getVaryingCounts());
		while ((curTile = renderer->fetchUnresolvedTileFragmentShading()) != -1) {
			for (int i = 0; i < context->numThreads; i++) {
				for (int j = 0; j < context->coverQueue[i][curTile].size(); j++) {
					auto& proposal = context->coverQueue[i][curTile][j];
					if (proposal.level == TileRasterLevel::PIXEL) {
						pixelShading(proposal.primitiveId, proposal.tile.x, proposal.tile.y);
					}
					else if (proposal.level == TileRasterLevel::TILE) {
						auto curTileX = curTile % context->tileBlocksX;
						auto curTileY = curTile / context->tileBlocksX;
						auto curTileX2 = (curTileX + 1) * frameBufferWidth / context->tileBlocksX;
						auto curTileY2 = (curTileY + 1) * frameBufferHeight / context->tileBlocksX;
						curTileX = curTileX * frameBufferWidth/ context->tileBlocksX;
						curTileY = curTileY * frameBufferHeight/ context->tileBlocksX;
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
	void TileRasterWorker::pixelShading(const int primitiveId, const int dx, const int dy) {
		int idx = primitiveId * context->vertexStride;
		float4 pos[3];
		pos[0] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx]];
		pos[1] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx + 1]];
		pos[2] = context->vertexShaderResult->getPositionBuffer()[(*context->indexBuffer)[idx + 2]];
		for (int i = 0; i < 3; i++) {
			pos[i].x /= pos[i].w;
			pos[i].y /= pos[i].w;
			pos[i].z /= pos[i].w;
		}
		
		float pDx = 2.0f * dx / context->frameBuffer->getColorAttachment(0)->getWidth() - 1.0f;
		float pDy = 2.0f * dy / context->frameBuffer->getColorAttachment(0)->getHeight() - 1.0f;
		float4 p = { pDx,pDy,1.0,1.0 };

		float bary[3], area;
		area = edgeFunction(pos[0], pos[1], pos[2]);
		bary[0] = edgeFunction(pos[1], pos[2], p)/area;
		bary[1] = edgeFunction(pos[2], pos[0], p)/area;
		bary[2] = edgeFunction(pos[0], pos[1], p)/area;
		
		float zCorr = 1.0 / (bary[0] / pos[0].w + bary[1] / pos[1].w + bary[2] / pos[2].w);

		// Interpolate Depth
		float depth[3];
		for (int i = 0; i < 3; i++) {
			depth[i] = pos[i].z / pos[i].w / pos[i].w;
		}
		float interpolatedDepth = bary[0] * depth[0] + bary[1] * depth[1] + bary[2] * depth[2];
		interpolatedDepth *= zCorr;

		// Depth Test
		auto& depthAttachment = *context->frameBuffer->getDepthAttachment();
		if (interpolatedDepth > depthAttachment(dx, dy, 0)) {
			return;
		}
		depthAttachment(dx, dy, 0) = interpolatedDepth;

		// Interpolate Varyings
		
		for (int i = 0; i < context->vertexShader->getVaryingCounts(); i++) {
			float w[3] = { pos[0].w,pos[1].w,pos[2].w };
			interpolatedVaryings[i] = interpolateVaryings(i, (*context->indexBuffer).data() + idx, { bary[0],bary[1],bary[2] }, zCorr, w);
		}

		// Fragment Shader
		context->fragmentShader->execute(interpolatedVaryings, colorOutput);
		context->frameBuffer->getColorAttachment(0)->fillAreaRGBA(dx, dy, 1, 1, colorOutput[0].x, colorOutput[0].y, colorOutput[0].z, colorOutput[0].w);
		

	}
	std::any TileRasterWorker::interpolateVaryings(int id,const int indices[3], const float4& barycentric, const float zCorr, const float w[3]) {
		auto varying = context->vertexShaderResult->getVaryingBuffer<char*>(id);
		auto varyingDescriptor = context->vertexShaderResult->getVaryingDescriptor(id);
		float bary[3]={barycentric.x,barycentric.y,barycentric.z};
		if (varyingDescriptor.type == TypeDescriptorEnum::FLOAT4) {
			float4* f4 = reinterpret_cast<float4*>(varying);
			float4 interpolated = { 0,0,0,0 };
			for (int j = 0; j < 3; j++) {
				interpolated.x += f4[indices[j]].x * bary[j] * zCorr / w[j];
				interpolated.y += f4[indices[j]].y * bary[j] * zCorr / w[j];
				interpolated.z += f4[indices[j]].z * bary[j] * zCorr / w[j];
				interpolated.w += f4[indices[j]].w * bary[j] * zCorr / w[j];
			}
			return interpolated;
		}else if (varyingDescriptor.type == TypeDescriptorEnum::FLOAT3){
			float3* f3 = reinterpret_cast<float3*>(varying);
			float3 interpolated3 = { 0,0,0 };
			for (int j = 0; j < 3; j++) {
				interpolated3.x += f3[indices[j]].x * bary[j] * zCorr / w[j];
				interpolated3.y += f3[indices[j]].y * bary[j] * zCorr / w[j];
				interpolated3.z += f3[indices[j]].z * bary[j] * zCorr / w[j];
			}
			return interpolated3;
		} else if (varyingDescriptor.type == TypeDescriptorEnum::FLOAT2){
			float2* f2 = reinterpret_cast<float2*>(varying);
			float2 interpolated2 = { 0,0 };
			for (int j = 0; j < 3; j++) {
				interpolated2.x += f2[indices[j]].x * bary[j] * zCorr / w[j];
				interpolated2.y += f2[indices[j]].y * bary[j] * zCorr / w[j];
			}
			return interpolated2;
		}
		else if (varyingDescriptor.type == TypeDescriptorEnum::FLOAT1) {
			float* f1 = reinterpret_cast<float*>(varying);
			float interpolated1 = 0;
			for (int j = 0; j < 3; j++) {
				interpolated1 += f1[indices[j]] * bary[j] * zCorr / w[j];
			}
			return interpolated1;
		}
		else {
			ifritError("Unsupported Varying Type");
			return {};
		}
	}



}