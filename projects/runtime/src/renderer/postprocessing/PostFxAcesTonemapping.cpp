#pragma once
#include "ifrit/runtime/renderer/postprocessing/PostFxAcesTonemapping.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

namespace Ifrit::Runtime::PostprocessPassCollection
{
    IFRIT_APIDECL PostFxAcesToneMapping::PostFxAcesToneMapping(IApplication* app)
        : PostprocessPass(app, { Internal::kIntShaderTable.Postprocess.ACESFS, 1, 1 })
    {
    }

    IFRIT_APIDECL void PostFxAcesToneMapping::RenderPostFx(
        const GPUCmdBuffer* cmd, RenderTargets* renderTargets, SRVDesc inputTexCombSampler)
    {
        struct PushConst
        {
            u32 inputTexCombSampler;
        };
        PushConst pushConst = { inputTexCombSampler };
        RenderInternal(nullptr, renderTargets, cmd, &pushConst, {}, "Postprocess: Aces Tone Mapping");
    }

} // namespace Ifrit::Runtime::PostprocessPassCollection