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