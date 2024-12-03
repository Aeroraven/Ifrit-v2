
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
#include "TrivialRaytracer.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/RaytracerBase.h"
#include <stack>

namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer {
struct RaytracingShaderGlobalVarSection {
  ShaderBase *shader;
};

class IFRIT_APIDECL TrivialRaytracerWorker {
private:
  int workerId;
  std::atomic<TrivialRaytracerWorkerStatus> status;
  TrivialRaytracer *renderer;
  std::shared_ptr<TrivialRaytracerContext> context;
  std::unique_ptr<std::thread> thread;

  std::stack<RaytracingShaderGlobalVarSection> execStack;
  int recurDepth = 0;

public:
  friend class TrivialRaytracer;
  TrivialRaytracerWorker(std::shared_ptr<TrivialRaytracer> renderer,
                         std::shared_ptr<TrivialRaytracerContext> context,
                         int workerId);
  void run();
  void threadCreate();

  void tracingProcess();
  void tracingRecursiveProcess(const RayInternal &ray, void *payload, int depth,
                               float tmin, float tmax);

  int getTracingDepth();
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer