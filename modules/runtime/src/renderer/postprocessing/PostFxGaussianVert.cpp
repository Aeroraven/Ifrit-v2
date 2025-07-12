#pragma once
#include "ifrit/runtime/renderer/postprocessing/PostFxGaussianVert.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

namespace Ifrit::Runtime::PostprocessPassCollection
{
    IFRIT_APIDECL PostFxGaussianVert::PostFxGaussianVert(IApplication* app)
        : PostprocessPass(app, { Internal::kIntShaderTable.Postprocess.GaussianVertFS, 2, 1 })
    {
    }

    IFRIT_APIDECL void PostFxGaussianVert::RenderPostFx(
        const GPUCmdBuffer* cmd, RenderTargets* renderTargets, SRVDesc inputTexCombSampler, u32 kernelSize)
    {
        struct PushConst
        {
            u32 inputTexCombSampler;
            u32 kernelSize;
        };
        PushConst pushConst = {
            inputTexCombSampler,
            kernelSize,
        };
        RenderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Vertical Gaussian Blur");
    }

} // namespace Ifrit::Runtime::PostprocessPassCollection