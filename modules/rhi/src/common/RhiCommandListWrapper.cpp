#include "ifrit/rhi/common/RhiCommandListWrapper.h"
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::RHI
{
    IFRIT_RHI_API void RhiCommandListBase::FinishRecording()
    {
        // TODO
    }

    IFRIT_RHI_API void RhiCommandListBase::EnqueueRHICommand(Owner<RhiCommand> cmd)
    {
        ScopedLock lock(mLock);
        IF_LOG_ASSERTION("RhiCommandListBase", cmd != nullptr, "Command cannot be null");
        IF_LOG_ASSERTION("RhiCommandListBase", mState == ERhiCommandListState::Recording,
            "Command must be in recording state before enqueueing");
        mCommands.push_back(std::move(cmd));
    }

    // Actual Cmds

    IFRIT_RHI_API void RhiCommandListBase::DispatchComputeShader(u32 groupX, u32 groupCountY, u32 groupCountZ)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_Dispatch>(groupX, groupCountY, groupCountZ));
    }
    IFRIT_RHI_API void RhiCommandListBase::DispatchComputeShaderIndirect(const RhiBuffer* buffer, u32 offset)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_DispatchIndirect>(buffer, offset));
    }

    IFRIT_RHI_API void RhiCommandListBase::Draw(u32 numPrimitives, u32 instanceCount, u32 firstVertex)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_DrawPrimitives>(numPrimitives, instanceCount, firstVertex));
    }

    IFRIT_RHI_API void RhiCommandListBase::DrawIndexed(RhiBuffer* indexBuffer, u32 baseVertexIndex, u32 firstInstance,
        u32 startIndex, u32 numIndices, u32 numInstances)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_DrawPrimitivesIndexed>(
            indexBuffer, baseVertexIndex, firstInstance, startIndex, numIndices, numInstances));
    }

    IFRIT_RHI_API void RhiCommandListBase::DispatchMeshShader(u32 groupCountX, u32 groupCountY, u32 groupCountZ)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_DrawMeshTasks>(groupCountX, groupCountY, groupCountZ));
    }
    IFRIT_RHI_API void RhiCommandListBase::DispatchMeshShaderIndirect(const RhiBuffer* buffer, u32 offset)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_DrawMeshTasksIndirect>(buffer, offset, 1, 0));
    }
    IFRIT_RHI_API void RhiCommandListBase::SetViewports(const Vec<RhiViewport>& viewport)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_SetViewport>(viewport));
    }
    IFRIT_RHI_API void RhiCommandListBase::SetScissors(const Vec<RhiScissor>& scissor)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_SetScissor>(scissor));
    }

} // namespace Ifrit::RHI