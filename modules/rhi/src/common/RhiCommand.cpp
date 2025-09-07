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

    // ===== Transition Commands =====
    void RhiCmd_BeginTransitions::Execute(RhiCommandListBase* cmd)
    {
        cmd->GetActiveContext()->CmdBeginTransitionList(mTransitions);
    }
    void RhiCmd_EndTransitions::Execute(RhiCommandListBase* cmd)
    {
        cmd->GetActiveContext()->CmdEndTransitionList(mTransitions);
    }

} // namespace Ifrit::RHI