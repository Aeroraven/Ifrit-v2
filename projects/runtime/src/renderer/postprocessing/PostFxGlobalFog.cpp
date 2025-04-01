#pragma once
#include "ifrit/runtime/renderer/postprocessing/PostFxGlobalFog.h"
#include "ifrit/core/typing/Util.h"

namespace Ifrit::Runtime::PostprocessPassCollection
{

    IFRIT_APIDECL PostFxGlobalFog::PostFxGlobalFog(IApplication* app)
        : PostprocessPass(app, { "GlobalFog.frag.glsl", 3, 1 })
    {
    }

    IFRIT_APIDECL void PostFxGlobalFog::RenderPostFx(const GPUCmdBuffer* cmd, RenderTargets* renderTargets,
        GPUBindId* inputTexCombSampler, GPUBindId* inputDepthTexCombSampler, GPUBindId* inputViewUniform)
    {
        struct PushConst
        {
            u32 inputTexCombSampler;
            u32 inputDepthTexCombSampler;
            u32 inputViewUniform;
        };
        PushConst pushConst = { inputTexCombSampler->GetActiveId(), inputDepthTexCombSampler->GetActiveId(),
            inputViewUniform->GetActiveId() };
        RenderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Global Fog");
    }

} // namespace Ifrit::Runtime::PostprocessPassCollection