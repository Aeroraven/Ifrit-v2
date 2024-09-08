#pragma once
#include "engine/base/Shaders.h"
#include "engine/base/RaytracerBase.h"
#include "core/data/Image.h"

namespace Ifrit::Engine::Raytracer {
	struct TrivialRaytracerContext {
		constexpr static int numThreads = 1;

		constexpr static int tileWidth = 16;
		constexpr static int tileHeight = 16;
		constexpr static int tileDepth = 1;
		int numTileX, numTileY, numTileZ;
		int totalTiles;

		RayGenShader* raygenShader;
		MissShader* missShader;
		CloseHitShader* closestHitShader;
		CallableShader* callableShader;
		const AccelerationStructure* accelerationStructure;

		iint3 traceRegion;

		Ifrit::Core::Data::ImageF32* testImage;
	};
}