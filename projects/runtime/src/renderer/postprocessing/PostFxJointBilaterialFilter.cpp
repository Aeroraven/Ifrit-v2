#pragma once
#include "ifrit/core/base/IfritBase.h"

#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/renderer/postprocessing/PostFxJointBilaterialFilter.h"

namespace Ifrit::Runtime::PostprocessPassCollection
{
    IFRIT_APIDECL
    PostFxJointBilaterialFilter::PostFxJointBilaterialFilter(IApplication* app)
        : PostprocessPass(app, { "JointBilaterialFilter.frag.glsl", 4, 1 })
    {
    }

    IFRIT_APIDECL void PostFxJointBilaterialFilter::RenderPostFx(const GPUCmdBuffer* cmd, RenderTargets* renderTargets,
        GPUBindId* colorSRV, GPUBindId* normalSRV, GPUBindId* depthSRV, u32 kernelSize)
    {
        struct PushConst
        {
            u32 colorSRV;
            u32 normalSRV;
            u32 depthSRV;
            u32 halfKernSize;
        };
        PushConst pushConst = {
            colorSRV->GetActiveId(),
            normalSRV->GetActiveId(),
            depthSRV->GetActiveId(),
            kernelSize,
        };
        RenderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Joint Bilaterial Filter");
    }

} // namespace Ifrit::Runtime::PostprocessPassCollection