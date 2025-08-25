#pragma once

#include "RhiBaseTypes.h"
#include "RhiApi.h"
#include "ifrit/rhi/common/RhiCommand.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/rhi/common/RhiDevice.h"

namespace Ifrit::RHI
{
    class RhiCommandListBase;
    class RhiCommandListExecutor;

    // ===== Command List Context =====
    class IFRIT_RHI_API RhiCommandListContext : public RhiDeviceChild
    {
    public:
        virtual ~RhiCommandListContext() = default;

    private:
        Vec<Ref<RhiTaskSubmission>> mRhiWaitSubmissions;
        ERhiCommandListPipelineType mRhiPipeline = ERhiCommandListPipelineType::Invalid;
        ERhiCommandListState        mRhiState    = ERhiCommandListState::Invalid;
        Mutex                       mRhiLock;
    };

    // ===== Command List Base (Recording) =====
    class IFRIT_RHI_API RhiCommandListBase : public RhiDeviceChild
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

        // Lambda
        void ExecuteLambda(Fn<void(const RhiCommandListBase* cmd)>&& lambda);

        // Utility
        void FinishRecording();
        RhiCommandListContext* GetActiveContext();
        RhiCommandListContext* GetUploadContext();
        void                   SwitchPipeline(ERhiCommandListPipelineType type);

    protected:
        void                          EnqueueRHICommand(Owner<RhiCommand> cmd);

        inline RhiCommandListContext* InternalGetActiveContext() { return mActiveContext.get(); }
        inline RhiCommandListContext* InternalGetUploadContext() { return mUploadContext.get(); }

        void                          AcquireActiveContext();
        void                          AcquireUploadContext();

        void                          ReleaseActiveContext();
        void                          ReleaseUploadContext();

        void                          SubmitActiveContext();
        void                          SubmitUploadContext();

        void                          AddPrerequisiteSubmission(Ref<RhiTaskSubmission> submission);

    protected:
        Owner<RhiCommandListContext> mActiveContext;
        Owner<RhiCommandListContext> mUploadContext;
        ERhiCommandListPipelineType  mRhiPipeline = ERhiCommandListPipelineType::Invalid;
        Vec<Owner<RhiCommand>>       mCommands;
        Vec<Ref<RhiTaskSubmission>>  mWaitSubmissions;
        bool                         mImmediateCmdList = false;
        bool                         mValid            = false;

        friend class RhiCommandListExecutor;
    };

    // ===== Command List Executor =====
    class IFRIT_RHI_API RhiCommandListExecutor : public RhiDeviceChild, public NonCopyable
    {
    public:
        RhiCommandListBase* CreateCommandList(RhiCommandListBase* immediateCmdList = nullptr);
        RhiCommandListBase* GetImmediateCommandList();

    private:
        RhiCommandListBase* mImmediateCmdList = nullptr;
    };
} // namespace Ifrit::RHI