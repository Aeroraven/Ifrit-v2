#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/RaytracerBase.h"
#include "TrivialRaytracer.h"

namespace Ifrit::Engine::Raytracer {

	class IFRIT_APIDECL TrivialRaytracerWorker {
	private:
		int workerId;
		std::atomic<TrivialRaytracerWorkerStatus> status;
		std::weak_ptr<TrivialRaytracer> renderer;
		std::shared_ptr<TrivialRaytracerContext> context;
		std::unique_ptr<std::thread> thread;

	public:
		friend class TrivialRaytracer;
		TrivialRaytracerWorker(std::shared_ptr<TrivialRaytracer> renderer, std::shared_ptr<TrivialRaytracerContext> context, int workerId);
		void run();
		void threadCreate();

		void tracingProcess();
	};
}