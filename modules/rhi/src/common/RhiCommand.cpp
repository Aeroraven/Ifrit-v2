#include "ifrit/rhi/common/RhiCommand.h"
#include "ifrit/rhi/common/RhiCommandList.h"
namespace Ifrit::RHI
{
    // ===== Pipeline State Commands =====
    void RhiCmd_SetComputePipelineState::Execute(RhiCommandListBase* cmd)
    {
        cmd->GetActiveContext()->CmdSetComputePipelineState(mDesc);
    }

    void RhiCmd_SetGraphicsPipelineState::Execute(RhiCommandListBase* cmd)
    {
        cmd->GetActiveContext()->CmdSetGraphicsPipelineState(mDesc);
    }

    void RhiCmd_SetShaderParameters::Execute(RhiCommandListBase* cmd)
    {
        cmd->GetActiveContext()->CmdSetShaderParameters(mParams);
    }

    // ===== Transition Commands =====
    void RhiCmd_BeginTransitions::Execute(RhiCommandListBase* cmd)
    {
        cmd->GetActiveContext()->CmdBeginTransitionList(mTransitions);
    }
    void RhiCmd_EndTransitions::Execute(RhiCommandListBase* cmd)
    {
        cmd->GetActiveContext()->CmdEndTransitionList(mTransitions);
    }

    // ===== Draw Calls =====
    void RhiCmd_Dispatch::Execute(RhiCommandListBase* cmd)
    {
        cmd->GetActiveContext()->CmdDispatch(mGroupCountX, mGroupCountY, mGroupCountZ);
    }

} // namespace Ifrit::RHI