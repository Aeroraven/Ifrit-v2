
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#pragma once
#include "RtShaders.h"
#include "ifrit/softgraphics/core/data/Image.h"
#include "ifrit/softgraphics/engine/base/RaytracerBase.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/raytracer/accelstruct/RtBoundingVolumeHierarchy.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer {
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
      Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::PairHash>
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

  Ifrit::GraphicsBackend::SoftGraphics::Core::Data::ImageF32 *testImage;
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer