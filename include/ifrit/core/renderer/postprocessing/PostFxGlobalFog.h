#pragma once
#include "ifrit/core/renderer/PostprocessPass.h"

namespace Ifrit::Core::PostprocessPassCollection {

class IFRIT_APIDECL PostFxGlobalFog : public PostprocessPass {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;

public:
  PostFxGlobalFog(IApplication *app);
  void renderPostFx(const GPUCmdBuffer *cmd, RenderTargets *renderTargets,
                    GPUBindId *inputTexCombSampler,
                    GPUBindId *inputDepthTexCombSampler,
                    GPUBindId *inputViewUniform);
};

} // namespace Ifrit::Core::PostprocessPassCollection