#include "engine/raytracer/TrivialRaytracerWorker.h"
#include "math/VectorOps.h"
namespace Ifrit::Engine::Raytracer {
	TrivialRaytracerWorker::TrivialRaytracerWorker(std::shared_ptr<TrivialRaytracer> renderer, std::shared_ptr<TrivialRaytracerContext> context, int workerId) {
		this->renderer = renderer.get();
		this->context = context;
		this->workerId = workerId;
	}
	void TrivialRaytracerWorker::run() {
		while (true) {
			const auto st = status.load();
			if (st == TrivialRaytracerWorkerStatus::IDLE || st == TrivialRaytracerWorkerStatus::COMPLETED) {
				std::this_thread::yield();
			}
			else if (st == TrivialRaytracerWorkerStatus::TERMINATED) {
				return;
			}
			else if (st == TrivialRaytracerWorkerStatus::TRACING) {
				tracingProcess();
				status.store(TrivialRaytracerWorkerStatus::TRACING_SYNC, std::memory_order::memory_order_relaxed);
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
	void TrivialRaytracerWorker::tracingRecursiveProcess(Ray ray, void* payload, int depth, float tmin, float tmax){
		using namespace Ifrit::Math;

		if (depth >= context->maxDepth)return;
		RayInternal intray;
		intray.o = ray.o;
		intray.r = ray.r;
		intray.invr = ifloat3{ 1.0f,1.0f,1.0f } / intray.r;
		auto collresult = context->accelerationStructure->queryIntersection(intray, tmin,tmax);
		recurDepth++;
		if (collresult.id == -1) {
			if (context->missShader) {
				context->perWorkerMiss[workerId]->pushStack(ray, collresult, payload);
				context->perWorkerMiss[workerId]->execute(this);
				context->perWorkerMiss[workerId]->popStack();
			}
		}
		else {
			context->perWorkerRayhit[workerId]->pushStack(ray, collresult, payload);
			context->perWorkerRayhit[workerId]->execute(collresult, ray, this);
			context->perWorkerRayhit[workerId]->popStack();
		}
		recurDepth--;
	}
	int TrivialRaytracerWorker::getTracingDepth(){
		return recurDepth;
	}
}
