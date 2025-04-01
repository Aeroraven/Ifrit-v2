#pragma once
#include "ifrit/runtime/renderer/postprocessing/PostFxGaussianHori.h"
#include "ifrit/core/typing/Util.h"

namespace Ifrit::Runtime::PostprocessPassCollection
{
    IFRIT_APIDECL PostFxGaussianHori::PostFxGaussianHori(IApplication* app)
        : PostprocessPass(app, { "GaussianVert.frag.glsl", 2, 1 })
    {
    }

    IFRIT_APIDECL void PostFxGaussianHori::RenderPostFx(
        const GPUCmdBuffer* cmd, RenderTargets* renderTargets, GPUBindId* inputTexCombSampler, u32 kernelSize)
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
        RenderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Horizontal Gaussian Blur");
    }

} // namespace Ifrit::Runtime::PostprocessPassCollection