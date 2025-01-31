#pragma once
#include "ifrit/common/base/IfritBase.h"

#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/renderer/postprocessing/PostFxJointBilaterialFilter.h"

namespace Ifrit::Core::PostprocessPassCollection {
IFRIT_APIDECL
PostFxJointBilaterialFilter::PostFxJointBilaterialFilter(IApplication *app)
    : PostprocessPass(app, {"JointBilaterialFilter.frag.glsl", 4, 1}) {}

IFRIT_APIDECL void PostFxJointBilaterialFilter::renderPostFx(
    const GPUCmdBuffer *cmd, RenderTargets *renderTargets, GPUBindId *colorSRV,
    GPUBindId *normalSRV, GPUBindId *depthSRV, uint32_t kernelSize) {
  struct PushConst {
    u32 colorSRV;
    u32 normalSRV;
    u32 depthSRV;
    u32 halfKernSize;
  };
  PushConst pushConst = {
      colorSRV->getActiveId(),
      normalSRV->getActiveId(),
      depthSRV->getActiveId(),
      kernelSize,
  };
  renderInternal(nullptr, renderTargets, cmd, &pushConst, {},
                 "Postprocess: Joint Bilaterial Filter");
}

} // namespace Ifrit::Core::PostprocessPassCollection