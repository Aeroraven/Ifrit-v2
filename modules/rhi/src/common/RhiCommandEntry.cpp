#include "ifrit/rhi/common/RhiCommandEntry.h"
#include "ifrit/rhi/common/RhiCommandListContext.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/rhi/common/RhiDevice.h"
#include "ifrit/rhi/common/RhiPipeline.h"
#include "ifrit/rhi/utils/RhiCmdTranslationHelper.h"

namespace Ifrit::RHI
{
    // Memory Transfer Commands
    void RhiCmd_CopyBuffer::Execute(const RhiCommandListContext* cmdCtx)
    {
        if (!mSrcBuffer || !mDstBuffer) IF_UNLIKELY
        {
            IF_LOG_CRITICAL("RhiCmd.CopyBuffer", "Source or destination buffer is null");
        }
        cmdCtx->CopyBuffer(mSrcBuffer, mDstBuffer, mSize, mSrcOffset, mDstOffset);
    }

    // Execution Commands
    IFRIT_APIDECL void RhiCmd_Dispatch::Execute(const RhiCommandListContext* cmdCtx)
    {
        if (mGroupCountX == 0 || mGroupCountY == 0 || mGroupCountZ == 0) IF_UNLIKELY
        {
            IF_LOG_WARNING("RhiCmd.Dispatch", "Group counts must be greater than zero");
            return;
        }
        cmdCtx->Dispatch(mGroupCountX, mGroupCountY, mGroupCountZ);
    }
    IFRIT_APIDECL void RhiCmd_DispatchIndirect::Execute(const RhiCommandListContext* cmdCtx)
    {
        if (!mBuffer) IF_UNLIKELY
        {
            IF_LOG_CRITICAL("RhiCmd.DispatchIndirect", "Buffer is null");
            return;
        }
        cmdCtx->DispatchIndirect(mBuffer, mOffset);
    }

    IFRIT_APIDECL void RhiCmd_DrawMeshTasks::Execute(const RhiCommandListContext* cmdCtx)
    {
        auto device = cmdCtx->GetContext();
        if (device->GetCapabilities().m_MeshShaderEnabled == false) IF_UNLIKELY
        {
            IF_LOG_CRITICAL("RhiCmd.DrawMeshTasks", "Mesh shaders are not enabled on this device");
            return;
        }
        if (mGroupCountX == 0 || mGroupCountY == 0 || mGroupCountZ == 0) IF_UNLIKELY
        {
            IF_LOG_WARNING("RhiCmd.DrawMeshTasks", "Group counts must be greater than zero");
            return;
        }
        cmdCtx->DrawMeshTasks(mGroupCountX, mGroupCountY, mGroupCountZ);
    }
    IFRIT_APIDECL void RhiCmd_DrawMeshTasksIndirect::Execute(const RhiCommandListContext* cmdCtx)
    {
        if (!mBuffer) IF_UNLIKELY
        {
            IF_LOG_CRITICAL("RhiCmd.DrawMeshTasksIndirect", "Buffer is null");
            return;
        }
        cmdCtx->DrawMeshTasksIndirect(mBuffer, mOffset, mDrawCount, mStride);
    }
    IFRIT_APIDECL void RhiCmd_DrawPrimitives::Execute(const RhiCommandListContext* cmdCtx)
    {
        auto gfx = cmdCtx->GetBoundGraphicsPipeline();
        IF_LOG_ASSERTION("RhiCmd.DrawPrimitives", gfx != nullptr, "No graphics pipeline bound");

        auto numVertices = GetNumVerticesFromPrimitive(gfx->GetRasterizerTopology(), mPrimCount);
        if (numVertices == 0) IF_UNLIKELY
        {
            IF_LOG_WARNING("RhiCmd.Draw", "Vertex count must be greater than zero");
            return;
        }
        cmdCtx->Draw(numVertices, mInstanceCount, mFirstVertex, 0);
    }

    IFRIT_APIDECL void RhiCmd_DrawPrimitivesIndexed::Execute(const RhiCommandListContext* cmdCtx)
    {
        if (mIndexBuffer == nullptr) IF_UNLIKELY
        {
            IF_LOG_CRITICAL("RhiCmd.DrawPrimitivesIndexed", "Index buffer is null");
            return;
        }
        cmdCtx->AttachIndexBuffer(mIndexBuffer);
        cmdCtx->DrawIndexed(mNumIndices, mNumInstances, mStartIndex, mBaseVertexIndex, mFirstInstance);
    }

    // Graphics Pass Specific Commands
    IFRIT_APIDECL void RhiCmd_SetViewport::Execute(const RhiCommandListContext* cmdCtx)
    {
        if (mViewports.empty()) IF_UNLIKELY
        {
            IF_LOG_WARNING("RhiCmd.SetViewport", "No viewports provided");
            return;
        }
        cmdCtx->SetViewports(mViewports);
    }
    IFRIT_APIDECL void RhiCmd_SetScissor::Execute(const RhiCommandListContext* cmdCtx)
    {
        if (mScissors.empty()) IF_UNLIKELY
        {
            IF_LOG_WARNING("RhiCmd.SetScissor", "No scissors provided");
            return;
        }
        cmdCtx->SetScissors(mScissors);
    }

} // namespace Ifrit::RHI