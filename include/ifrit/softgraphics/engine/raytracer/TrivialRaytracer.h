
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
#include "TrivialRaytracerContext.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/softgraphics/core/data/Image.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/RaytracerBase.h"
#include "ifrit/softgraphics/engine/base/Renderer.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/bufferman/BufferManager.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer {
class TrivialRaytracerWorker;

enum class TrivialRaytracerWorkerStatus { IDLE, TRACING, TRACING_SYNC, COMPLETED, TERMINATED };

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
  void updateUniformBuffer();

public:
  friend class TrivialRaytracerWorker;
  TrivialRaytracer();
  ~TrivialRaytracer();
  void init();
  void bindAccelerationStructure(const BoundingVolumeHierarchyTopLevelAS *as);
  void bindRaygenShader(RayGenShader *shader);
  void bindMissShader(MissShader *shader);
  void bindClosestHitShader(CloseHitShader *shader);
  void bindCallableShader(CallableShader *shader);
  void bindUniformBuffer(int binding, int set, BufferManager::IfritBuffer pBuffer);

  void traceRays(u32 width, u32 height, u32 depth);

  void bindTestImage(Ifrit::GraphicsBackend::SoftGraphics::Core::Data::ImageF32 *image);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer