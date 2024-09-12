#include "engine/raytracer/TrivialRaytracer.h"
#include "engine/raytracer/TrivialRaytracerWorker.h"

namespace Ifrit::Engine::Raytracer {
	TrivialRaytracer::TrivialRaytracer() {
		
	}
	TrivialRaytracer::~TrivialRaytracer() {
		if (initialized) {
			for (auto& worker : workers) {
				worker->status.store(TrivialRaytracerWorkerStatus::TERMINATED, std::memory_order::memory_order_relaxed);
			}
			for (auto& worker : workers) {
				worker->thread->join();
			}
			initialized = false;
		}
		ifritLog1("TrivialRaytracer finalized");
	}
	void TrivialRaytracer::init() {
		initialized = true;
		context = std::make_shared<TrivialRaytracerContext>();
		context->perWorkerMiss.resize(context->numThreads);
		context->perWorkerRaygen.resize(context->numThreads);
		context->perWorkerRayhit.resize(context->numThreads);

		for (int i = 0; i < context->numThreads; i++) {
			auto worker = std::make_unique<TrivialRaytracerWorker>(shared_from_this(), context, i);
			worker->status.store(TrivialRaytracerWorkerStatus::IDLE, std::memory_order::memory_order_relaxed);
			worker->threadCreate();
			workers.push_back(std::move(worker));
		}
	}
	void TrivialRaytracer::bindAccelerationStructure(const AccelerationStructure* as) {
		context->accelerationStructure = as;
	}
	void TrivialRaytracer::bindRaygenShader(RayGenShader* shader) {
		context->raygenShader = shader;
		for (int i = 0; i < context->numThreads; i++) {
			context->perWorkerRaygen[i] = context->raygenShader->getThreadLocalCopy();
		}
	}
	void TrivialRaytracer::bindMissShader(MissShader* shader) {
		context->missShader = shader;
		for (int i = 0; i < context->numThreads; i++) {
			context->perWorkerMiss[i] = context->missShader->getThreadLocalCopy();
		}
	}
	void TrivialRaytracer::bindClosestHitShader(CloseHitShader* shader) {
		context->closestHitShader = shader;
		for (int i = 0; i < context->numThreads; i++) {
			context->perWorkerRayhit[i] = context->closestHitShader->getThreadLocalCopy();
		}
	}
	void TrivialRaytracer::bindCallableShader(CallableShader* shader) {
		context->callableShader = shader;
	}
	void TrivialRaytracer::bindUniformBuffer(int binding, int set, BufferManager::IfritBuffer pBuffer) {
		auto p = pBuffer.manager.lock();
		void* data;
		p->mapBufferMemory(pBuffer, &data);
		this->context->uniformMapping[{binding, set}] = data;
	}
	void TrivialRaytracer::traceRays(uint32_t width, uint32_t height, uint32_t depth) {
		context->traceRegion = iint3(width, height, depth);
		context->numTileX = (width + context->tileWidth - 1) / context->tileWidth;
		context->numTileY = (height + context->tileHeight - 1) / context->tileHeight;
		context->numTileZ = (depth + context->tileDepth - 1) / context->tileDepth;
		context->totalTiles = context->numTileX * context->numTileY * context->numTileZ;
		unresolvedTiles = context->totalTiles;
		updateUniformBuffer();
		for (auto& worker : workers) {
			worker->status.store(TrivialRaytracerWorkerStatus::TRACING, std::memory_order::memory_order_relaxed);
		}
		statusTransitionBarrier(TrivialRaytracerWorkerStatus::TRACING_SYNC, TrivialRaytracerWorkerStatus::COMPLETED);
	}
	int TrivialRaytracer::fetchUnresolvedTiles() {
		return unresolvedTiles.fetch_sub(1) - 1;
	}
	void TrivialRaytracer::updateUniformBuffer(){
		auto rgenUniforms = context->raygenShader->getUniformList();
		for (int i = 0; i < context->numThreads; i++) {
			for (const auto& x : rgenUniforms) {
				if (context->uniformMapping.count(x)) {
					context->perWorkerRaygen[i]->updateUniformData(x.first, x.second, context->uniformMapping[x]);
				}
			}
		}
	}
	void TrivialRaytracer::resetWorkers() {
		for (auto& worker : workers) {
			worker->status.store(TrivialRaytracerWorkerStatus::IDLE, std::memory_order::memory_order_relaxed);
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