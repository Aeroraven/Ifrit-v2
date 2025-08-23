#pragma once

#include "RhiBaseTypes.h"
#include "RhiApi.h"
#include "ifrit/rhi/common/RhiCommandEntry.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/core/typing/Util.h"

namespace Ifrit::RHI
{
    class RhiCommandListBase;
    class RhiCommandListAllocator;

    enum class ERhiCommandListType
    {
        Invalid,
        Graphics,
        Compute,
        Transfer,
        RayTracing,
    };

    enum class ERhiCommandListState
    {
        Invalid,
        Recording,
        Submitted,
    };

    class IFRIT_RHI_API RhiCommandListBase
    {
    public:
        // Compute Dispatch
        void DispatchComputeShader(u32 groupX, u32 groupCountY, u32 groupCountZ);
        void DispatchComputeShaderIndirect(const RhiBuffer* buffer, u32 offset);

        // Draw Calls
        void Draw(u32 numPrimitives, u32 instanceCount, u32 firstVertex);
        void DrawIndexed(RhiBuffer* indexBuffer, u32 baseVertexIndex, u32 firstInstance, u32 startIndex, u32 numIndices,
            u32 numInstances);
        void DispatchMeshShader(u32 groupCountX, u32 groupCountY, u32 groupCountZ);
        void DispatchMeshShaderIndirect(const RhiBuffer* buffer, u32 offset);

        // Transfer Ops

        // Viewports
        void SetViewports(const Vec<RhiViewport>& viewport);
        void SetScissors(const Vec<RhiScissor>& scissor);

        // Utility
        void FinishRecording();

    protected:
        void EnqueueRHICommand(Owner<RhiCommand> cmd);

    protected:
        Vec<RhiCommandListBase*> mPrerequisites;
        Vec<RhiCommandListBase*> mSubsequent;
        Vec<Owner<RhiCommand>>   mCommands;
        ERhiCommandListType      mType;
        ERhiCommandListState     mState;
        Mutex                    mLock;

        friend class RhiCommandListAllocator;
    };

    class IFRIT_RHI_API RhiCommandListAllocator : public NonCopyable
    {
    public:
        RhiCommandListBase* AllocateCommandList(ERhiCommandListType type, Vec<RhiCommandListBase*> prerequisites);

    protected:
        void SubmitCommandList(RhiCommandListBase* cmdList);

    private:
        Vec<Owner<RhiCommandListBase>> mCommandLists;

        friend class RhiCommandListContext;
    };

} // namespace Ifrit::RHI