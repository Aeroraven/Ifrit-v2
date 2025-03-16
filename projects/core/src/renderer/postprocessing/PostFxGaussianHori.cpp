#pragma once
#include "ifrit/core/renderer/postprocessing/PostFxGaussianHori.h"
#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Core::PostprocessPassCollection {
IFRIT_APIDECL PostFxGaussianHori::PostFxGaussianHori(IApplication *app)
    : PostprocessPass(app, {"GaussianVert.frag.glsl", 2, 1}) {}

IFRIT_APIDECL void PostFxGaussianHori::renderPostFx(const GPUCmdBuffer *cmd, RenderTargets *renderTargets,
                                                    GPUBindId *inputTexCombSampler, uint32_t kernelSize) {
  struct PushConst {
    uint32_t inputTexCombSampler;
    uint32_t kernelSize;
  };
  PushConst pushConst = {
      inputTexCombSampler->getActiveId(),
      kernelSize,
  };
  renderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Horizontal Gaussian Blur");
}

} // namespace Ifrit::Core::PostprocessPassCollection