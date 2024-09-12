#pragma once
#include "engine/base/Shaders.h"
#include "engine/base/RaytracerBase.h"
#include "core/data/Image.h"
#include "RtShaders.h"

namespace Ifrit::Engine::Raytracer {
	struct TrivialRaytracerContext {
		constexpr static int numThreads = 16;

		constexpr static int tileWidth = 32;
		constexpr static int tileHeight = 32;
		constexpr static int tileDepth = 1;
		constexpr static int maxDepth = 15;

		int numTileX, numTileY, numTileZ;
		int totalTiles;

		std::unordered_map<std::pair<int, int>, const void*, Ifrit::Core::Utility::PairHash> uniformMapping;

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