#pragma once
#include "ifrit/runtime/renderer/postprocessing/PostFxAcesTonemapping.h"
#include "ifrit/core/typing/Util.h"

namespace Ifrit::Runtime::PostprocessPassCollection
{
    static PostprocessPassConfig kAcesToneMappingConfig = {
        .fragPath          = "Postproc.AcesTonemapping.frag.glsl",
        .numPushConstants  = Ifrit::SizeCast<u32>(sizeof(float) * 2),
        .numDescriptorSets = 0,
    };
    IFRIT_APIDECL PostFxAcesToneMapping::PostFxAcesToneMapping(IApplication* app)
        : PostprocessPass(app, { "ACESToneMapping.frag.glsl", 1, 1 })
    {
    }

    IFRIT_APIDECL void PostFxAcesToneMapping::RenderPostFx(
        const GPUCmdBuffer* cmd, RenderTargets* renderTargets, GPUBindId* inputTexCombSampler)
    {
        struct PushConst
        {
            u32 inputTexCombSampler;
        };
        PushConst pushConst = { inputTexCombSampler->GetActiveId() };
        RenderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Aces Tone Mapping");
    }

} // namespace Ifrit::Runtime::PostprocessPassCollection