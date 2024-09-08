#include "engine/raytracer/TrivialRaytracerWorker.h"

namespace Ifrit::Engine::Raytracer {
	TrivialRaytracerWorker::TrivialRaytracerWorker(std::shared_ptr<TrivialRaytracer> renderer, std::shared_ptr<TrivialRaytracerContext> context, int workerId) {
		this->renderer = renderer;
		this->context = context;
		this->workerId = workerId;
	}
	void TrivialRaytracerWorker::run() {
		while (true) {
			if (status.load() == TrivialRaytracerWorkerStatus::IDLE || status.load() == TrivialRaytracerWorkerStatus::TERMINATED) {
				std::this_thread::yield();
			}
			if (status.load() == TrivialRaytracerWorkerStatus::TRACING) {
				tracingProcess();
				status.store(TrivialRaytracerWorkerStatus::TRACING_SYNC);
			}
		}
	}
	void TrivialRaytracerWorker::threadCreate() {
		thread = std::make_unique<std::thread>(&TrivialRaytracerWorker::run, this);
	}

	void TrivialRaytracerWorker::tracingProcess() {
		// Tracing process
		auto curTile = 0;
		auto rendererTemp = renderer.lock();
		while((curTile = rendererTemp->fetchUnresolvedTiles()) >= 0) {
			auto tileX = curTile % context->numTileX;
			auto tileY = (curTile / context->numTileX) % context->numTileY;
			auto tileZ = curTile / (context->numTileX * context->numTileY);
			
			for(int i=0;i<context->tileWidth;i++) {
				for(int j=0;j<context->tileHeight;j++) {
					for(int k=0;k<context->tileDepth;k++) {
						iint3 invocation = iint3(tileX * context->tileWidth + i, tileY * context->tileHeight + j, tileZ * context->tileDepth + k);
						
						// Demo
						Ray ray;
						float rx = 1.0f * (i + tileX * context->tileWidth) / context->traceRegion.x;
						float ry = 1.0f * (j + tileY * context->tileHeight) / context->traceRegion.y;
						float rz = 1.0f * (k + tileZ * context->tileDepth) / context->traceRegion.z;

						rx = 0.5f * rx - 0.25f;
						ry = 0.5f * ry - 0.25f;
						rz = -1.0;

						ray.o = { rx,ry,rz };
						ray.r = { 0.0f,0.0f,1.0f };

						auto result = context->accelerationStructure->queryIntersection(ray);
						if(result.id>=0) {
							ifloat4 color = {1.0f, 0.0f, 0.0f, 1.0f };
							context->testImage->fillPixelRGBA(invocation.x, invocation.y, 1, 0, 0, 0);
						}
						else {
							context->testImage->fillPixelRGBA(invocation.x, invocation.y, 0, 0, 1, 0);
							//printf("%f %f %f\n", rx, ry,rz);
						}

					}
				}
			}
		}
	}
}