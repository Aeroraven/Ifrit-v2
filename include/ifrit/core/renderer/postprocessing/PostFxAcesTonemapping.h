#pragma once
#include "ifrit/core/renderer/PostprocessPass.h"

namespace Ifrit::Core::PostprocessPassCollection {

class IFRIT_APIDECL PostFxAcesToneMapping : public PostprocessPass {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;

public:
  PostFxAcesToneMapping(IApplication *app);
  void renderPostFx(const GPUCmdBuffer *cmd, RenderTargets *renderTargets,
                    GPUBindId *inputTexCombSampler);
};

} // namespace Ifrit::Core::PostprocessPassCollection