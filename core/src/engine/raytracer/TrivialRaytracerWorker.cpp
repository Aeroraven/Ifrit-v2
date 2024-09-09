#include "engine/raytracer/TrivialRaytracerWorker.h"

namespace Ifrit::Engine::Raytracer {
	TrivialRaytracerWorker::TrivialRaytracerWorker(std::shared_ptr<TrivialRaytracer> renderer, std::shared_ptr<TrivialRaytracerContext> context, int workerId) {
		this->renderer = renderer.get();
		this->context = context;
		this->workerId = workerId;
	}
	void TrivialRaytracerWorker::run() {
		while (true) {
			if (status.load() == TrivialRaytracerWorkerStatus::IDLE || status.load() == TrivialRaytracerWorkerStatus::COMPLETED) {
				std::this_thread::yield();
			}
			if (status.load() == TrivialRaytracerWorkerStatus::TERMINATED) {
				return;
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
		auto rendererTemp = renderer;
		while((curTile = rendererTemp->fetchUnresolvedTiles()) >= 0) {
			auto tileX = curTile % context->numTileX;
			auto tileY = (curTile / context->numTileX) % context->numTileY;
			auto tileZ = curTile / (context->numTileX * context->numTileY);
			
			for(int i=0;i<context->tileWidth;i++) {
				if (tileX * context->tileWidth + i >= context->traceRegion.x) break;
				for(int j=0;j<context->tileHeight;j++) {
					if (tileY * context->tileHeight + j >= context->traceRegion.y) break;
					for(int k=0;k<context->tileDepth;k++) {
						if (tileZ * context->tileDepth + k >= context->traceRegion.z) break;
						iint3 invocation = iint3(tileX * context->tileWidth + i, tileY * context->tileHeight + j, tileZ * context->tileDepth + k);
						context->perWorkerRaygen[workerId]->execute(invocation, context->traceRegion, this);
					}
				}
			}
		}
	}
	void TrivialRaytracerWorker::tracingRecursiveProcess(Ray ray, void* payload, size_t payloadSize, int depth){
		if (depth >= context->maxDepth)return;
		auto collresult = context->accelerationStructure->queryIntersection(ray);
		recurDepth++;
		if (collresult.id == -1) {
			if (context->missShader) {
				context->perWorkerMiss[workerId]->pushStack(ray, collresult, payload);
				context->perWorkerMiss[workerId]->execute(ray, payload, this);
				context->perWorkerMiss[workerId]->popStack();
			}
		}
		else {
			context->perWorkerRayhit[workerId]->pushStack(ray, collresult, payload);
			context->perWorkerRayhit[workerId]->execute(collresult, ray, payload, this);
			context->perWorkerRayhit[workerId]->popStack();
		}
		recurDepth--;
	}
	int TrivialRaytracerWorker::getTracingDepth(){
		return recurDepth;
	}
}


/*

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
*/