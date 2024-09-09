#pragma once
#include "engine/base/Shaders.h"
#include "engine/base/RaytracerBase.h"
#include "core/data/Image.h"
#include "RtShaders.h"

namespace Ifrit::Engine::Raytracer {
	struct TrivialRaytracerContext {
		constexpr static int numThreads = 16;

		constexpr static int tileWidth = 16;
		constexpr static int tileHeight = 16;
		constexpr static int tileDepth = 5;
		constexpr static int maxDepth = 15;

		int numTileX, numTileY, numTileZ;
		int totalTiles;

		//TODO: Shader binding table & Shader groups
		RayGenShader* raygenShader;
		MissShader* missShader;
		CloseHitShader* closestHitShader;
		CallableShader* callableShader;

		std::vector<std::unique_ptr<RayGenShader>> perWorkerRaygen;
		std::vector<std::unique_ptr<CloseHitShader>> perWorkerRayhit;
		std::vector<std::unique_ptr<MissShader>> perWorkerMiss;

		const AccelerationStructure* accelerationStructure;

		iint3 traceRegion;

		Ifrit::Core::Data::ImageF32* testImage;
	};
}