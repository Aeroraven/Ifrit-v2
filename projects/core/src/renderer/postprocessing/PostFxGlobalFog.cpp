#pragma once
#include "ifrit/core/renderer/postprocessing/PostFxGlobalFog.h"
#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Core::PostprocessPassCollection {

IFRIT_APIDECL PostFxGlobalFog::PostFxGlobalFog(IApplication *app)
    : PostprocessPass(app, {"GlobalFog.frag.glsl", 3, 1}) {}

IFRIT_APIDECL void PostFxGlobalFog::renderPostFx(const GPUCmdBuffer *cmd, RenderTargets *renderTargets,
                                                 GPUBindId *inputTexCombSampler, GPUBindId *inputDepthTexCombSampler,
                                                 GPUBindId *inputViewUniform) {
  struct PushConst {
    uint32_t inputTexCombSampler;
    uint32_t inputDepthTexCombSampler;
    uint32_t inputViewUniform;
  };
  PushConst pushConst = {inputTexCombSampler->getActiveId(), inputDepthTexCombSampler->getActiveId(),
                         inputViewUniform->getActiveId()};
  renderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Global Fog");
}

} // namespace Ifrit::Core::PostprocessPassCollection