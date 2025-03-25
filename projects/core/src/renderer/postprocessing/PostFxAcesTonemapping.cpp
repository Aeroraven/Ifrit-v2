#pragma once
#include "ifrit/core/renderer/postprocessing/PostFxAcesTonemapping.h"
#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Core::PostprocessPassCollection {
static PostprocessPassConfig kAcesToneMappingConfig = {
    .fragPath = "Postproc.AcesTonemapping.frag.glsl",
    .numPushConstants = Ifrit::Common::Utility::size_cast<u32>(sizeof(float) * 2),
    .numDescriptorSets = 0,
};
IFRIT_APIDECL PostFxAcesToneMapping::PostFxAcesToneMapping(IApplication *app)
    : PostprocessPass(app, {"ACESToneMapping.frag.glsl", 1, 1}) {}

IFRIT_APIDECL void PostFxAcesToneMapping::renderPostFx(const GPUCmdBuffer *cmd, RenderTargets *renderTargets,
                                                       GPUBindId *inputTexCombSampler) {
  struct PushConst {
    u32 inputTexCombSampler;
  };
  PushConst pushConst = {inputTexCombSampler->getActiveId()};
  renderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Aces Tone Mapping");
}

} // namespace Ifrit::Core::PostprocessPassCollection