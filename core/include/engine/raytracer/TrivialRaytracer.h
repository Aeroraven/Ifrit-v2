#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/RaytracerBase.h"
#include "engine/base/Shaders.h"
#include "engine/base/Renderer.h"
#include "core/data/Image.h"
#include "TrivialRaytracerContext.h"

namespace Ifrit::Engine::Raytracer {
	class TrivialRaytracerWorker;

	enum class TrivialRaytracerWorkerStatus {
		IDLE,
		TRACING,
		TRACING_SYNC,
		COMPLETED,
		TERMINATED
	};

	class IFRIT_APIDECL TrivialRaytracer : public Renderer, public std::enable_shared_from_this<TrivialRaytracer> {
	private:
		std::shared_ptr<TrivialRaytracerContext> context;
		std::vector<std::unique_ptr<TrivialRaytracerWorker>> workers;
		std::atomic<int> unresolvedTiles;
		bool initialized = false;
	protected:
		void resetWorkers();
		void statusTransitionBarrier(TrivialRaytracerWorkerStatus waitOn, TrivialRaytracerWorkerStatus proceedTo);
		int fetchUnresolvedTiles();
	public:
		friend class TrivialRaytracerWorker;
		TrivialRaytracer();
		~TrivialRaytracer();
		void init();
		void bindAccelerationStructure(const AccelerationStructure* as);
		void bindRaygenShader(RayGenShader* shader);
		void bindMissShader(MissShader* shader);
		void bindClosestHitShader(CloseHitShader* shader);
		void bindCallableShader(CallableShader* shader);
		void traceRays(uint32_t width, uint32_t height, uint32_t depth);

		void bindTestImage(Ifrit::Core::Data::ImageF32* image);
	};
}