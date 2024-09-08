#include "engine/raytracer/TrivialRaytracer.h"
#include "engine/raytracer/TrivialRaytracerWorker.h"

namespace Ifrit::Engine::Raytracer {
	TrivialRaytracer::TrivialRaytracer() {
		
	}
	TrivialRaytracer::~TrivialRaytracer() = default;
	void TrivialRaytracer::init() {
		context = std::make_shared<TrivialRaytracerContext>();
		for (int i = 0; i < context->numThreads; i++) {
			auto worker = std::make_unique<TrivialRaytracerWorker>(shared_from_this(), context, i);
			worker->status.store(TrivialRaytracerWorkerStatus::IDLE);
			worker->threadCreate();
			workers.push_back(std::move(worker));
		}
	}
	void TrivialRaytracer::bindAccelerationStructure(const AccelerationStructure* as) {
		context->accelerationStructure = as;
	}
	void TrivialRaytracer::bindRaygenShader(RayGenShader* shader) {
		context->raygenShader = shader;
	}
	void TrivialRaytracer::bindMissShader(MissShader* shader) {
		context->missShader = shader;
	}
	void TrivialRaytracer::bindClosestHitShader(CloseHitShader* shader) {
		context->closestHitShader = shader;
	}
	void TrivialRaytracer::bindCallableShader(CallableShader* shader) {
		context->callableShader = shader;
	}
	void TrivialRaytracer::traceRays(uint32_t width, uint32_t height, uint32_t depth) {
		context->traceRegion = iint3(width, height, depth);
		context->numTileX = (width + context->tileWidth - 1) / context->tileWidth;
		context->numTileY = (height + context->tileHeight - 1) / context->tileHeight;
		context->numTileZ = (depth + context->tileDepth - 1) / context->tileDepth;
		context->totalTiles = context->numTileX * context->numTileY * context->numTileZ;
		unresolvedTiles = context->totalTiles;
		for (auto& worker : workers) {
			worker->status.store(TrivialRaytracerWorkerStatus::TRACING);
		}
		statusTransitionBarrier(TrivialRaytracerWorkerStatus::TRACING_SYNC, TrivialRaytracerWorkerStatus::TERMINATED);
	}
	int TrivialRaytracer::fetchUnresolvedTiles() {
		return unresolvedTiles.fetch_sub(1) - 1;
	}
	void TrivialRaytracer::resetWorkers() {
		for (auto& worker : workers) {
			worker->status.store(TrivialRaytracerWorkerStatus::IDLE);
		}
	}
	void TrivialRaytracer::statusTransitionBarrier(TrivialRaytracerWorkerStatus waitOn, TrivialRaytracerWorkerStatus proceedTo) {
		while (true) {
			bool allWorkersReady = true;
			for (auto& worker : workers) {
				if (worker->status.load() != waitOn) {
					allWorkersReady = false;
					break;
				}
			}
			if (allWorkersReady) {
				for (auto& worker : workers) {
					worker->status.store(proceedTo);
				}
				break;
			}
		}
	}
	void TrivialRaytracer::bindTestImage(Ifrit::Core::Data::ImageF32* image) {
		context->testImage = image;
	}
	
}