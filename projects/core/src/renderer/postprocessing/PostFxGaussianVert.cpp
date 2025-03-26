#pragma once
#include "ifrit/core/renderer/postprocessing/PostFxGaussianVert.h"
#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Core::PostprocessPassCollection
{
    IFRIT_APIDECL PostFxGaussianVert::PostFxGaussianVert(IApplication* app)
        : PostprocessPass(app, { "GaussianVert.frag.glsl", 2, 1 }) {}

    IFRIT_APIDECL void PostFxGaussianVert::RenderPostFx(const GPUCmdBuffer* cmd, RenderTargets* renderTargets,
        GPUBindId* inputTexCombSampler, u32 kernelSize)
    {
        struct PushConst
        {
            u32 inputTexCombSampler;
            u32 kernelSize;
        };
        PushConst pushConst = {
            inputTexCombSampler->GetActiveId(),
            kernelSize,
        };
        RenderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Vertical Gaussian Blur");
    }

} // namespace Ifrit::Core::PostprocessPassCollection