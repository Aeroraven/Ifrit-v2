#pragma once
#include "RtShaders.h"
#include "ifrit/softgraphics/core/data/Image.h"
#include "ifrit/softgraphics/engine/base/RaytracerBase.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/raytracer/accelstruct/RtBoundingVolumeHierarchy.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Raytracer {
struct TrivialRaytracerContext {
  constexpr static int numThreads = 16;

  constexpr static int tileWidth = 32;
  constexpr static int tileHeight = 32;
  constexpr static int tileDepth = 1;
  constexpr static int maxDepth = 15;

  int numTileX, numTileY, numTileZ;
  int totalTiles;

  std::unordered_map<
      std::pair<int, int>, const void *,
      Ifrit::Engine::GraphicsBackend::SoftGraphics::Core::Utility::PairHash>
      uniformMapping;

  // TODO: Shader binding table & Shader groups
  RayGenShader *raygenShader;
  MissShader *missShader;
  CloseHitShader *closestHitShader;
  CallableShader *callableShader;

  std::atomic<int> dcnt = 0;
  std::vector<std::unique_ptr<RayGenShader>> perWorkerRaygen;
  std::vector<std::unique_ptr<CloseHitShader>> perWorkerRayhit;
  std::vector<std::unique_ptr<MissShader>> perWorkerMiss;

  const BoundingVolumeHierarchyTopLevelAS *accelerationStructure;

  iint3 traceRegion;

  Ifrit::Engine::GraphicsBackend::SoftGraphics::Core::Data::ImageF32 *testImage;
};
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Raytracer