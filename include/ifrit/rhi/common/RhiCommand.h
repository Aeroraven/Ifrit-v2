#pragma once
#include "RhiApi.h"
#include "RhiBaseTypes.h"
#include "RhiForwardingTypes.h"
#include "RhiPipeline.h"
#include "RhiTransition.h"

namespace Ifrit::RHI
{
    struct IFRIT_RHI_API RhiCommand
    {
    public:
        virtual ~RhiCommand() noexcept                = default;
        virtual void Execute(RhiCommandListBase* cmd) = 0;
    };

#define DECLARE_RHI_COMMAND(name) struct IFRIT_RHI_API name final : public RhiCommand

    // Lambda Command
    DECLARE_RHI_COMMAND(RhiCmd_Lambda)
    {
        using LambdaType = Fn<void(RhiCommandListBase * cmd)>;
        LambdaType mLambda;

        RhiCmd_Lambda(LambdaType lambda) : mLambda(std::move(lambda)) {}
        void Execute(RhiCommandListBase * cmd) override final { mLambda(cmd); }
    };

    // Pipeline State Commands
    DECLARE_RHI_COMMAND(RhiCmd_SetComputePipelineState)
    {
        RhiComputePipelineStateDesc mDesc;

        RhiCmd_SetComputePipelineState(const RhiComputePipelineStateDesc& desc) : mDesc(desc) {}
        void Execute(RhiCommandListBase * cmd) override final;
    };

    DECLARE_RHI_COMMAND(RhiCmd_SetGraphicsPipelineState)
    {
        RhiGraphicsPipelineStateDesc mDesc;

        RhiCmd_SetGraphicsPipelineState(const RhiGraphicsPipelineStateDesc& desc) : mDesc(desc) {}
        void Execute(RhiCommandListBase * cmd) override final;
    };

    DECLARE_RHI_COMMAND(RhiCmd_SetShaderParameters)
    {
        RhiShaderParameter mParams;

        RhiCmd_SetShaderParameters(const RhiShaderParameter& params) : mParams(params) {}
        void Execute(RhiCommandListBase * cmd) override final;
    };

    // Transition Commands
    DECLARE_RHI_COMMAND(RhiCmd_BeginTransitions)
    {
        Vec<Ref<RhiTransition>> mTransitions;

        RhiCmd_BeginTransitions(const Vec<Ref<RhiTransition>>& transitions) : mTransitions(transitions) {}
        void Execute(RhiCommandListBase * cmd) override final;
    };

    DECLARE_RHI_COMMAND(RhiCmd_EndTransitions)
    {
        Vec<Ref<RhiTransition>> mTransitions;

        RhiCmd_EndTransitions(const Vec<Ref<RhiTransition>>& transitions) : mTransitions(transitions) {}
        void Execute(RhiCommandListBase * cmd) override final;
    };

    // Draw Calls
    DECLARE_RHI_COMMAND(RhiCmd_Dispatch)
    {
        u32 mGroupCountX;
        u32 mGroupCountY;
        u32 mGroupCountZ;

        RhiCmd_Dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ)
            : mGroupCountX(groupCountX), mGroupCountY(groupCountY), mGroupCountZ(groupCountZ)
        {
        }
        void Execute(RhiCommandListBase * cmd) override final;
    };

    // // Memory Transfer Commands
    // DECLARE_RHI_COMMAND(RhiCmd_CopyBuffer)
    // {
    //     RhiBuffer* mSrcBuffer;
    //     RhiBuffer* mDstBuffer;
    //     u32        mSize;
    //     u32        mSrcOffset;
    //     u32        mDstOffset;

    //     RhiCmd_CopyBuffer(RhiBuffer * srcBuffer, RhiBuffer * dstBuffer, u32 size, u32 srcOffset, u32 dstOffset)
    //         : mSrcBuffer(srcBuffer), mDstBuffer(dstBuffer), mSize(size), mSrcOffset(srcOffset), mDstOffset(dstOffset)
    //     {
    //     }
    //     void Execute(const RhiCommandListBase* cmd) override;
    // };

    // // Execution Commands
    // DECLARE_RHI_COMMAND(RhiCmd_Dispatch)
    // {
    //     u32 mGroupCountX;
    //     u32 mGroupCountY;
    //     u32 mGroupCountZ;

    //     RhiCmd_Dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ)
    //         : mGroupCountX(groupCountX), mGroupCountY(groupCountY), mGroupCountZ(groupCountZ)
    //     {
    //     }
    //     void Execute(const RhiCommandListBase* cmd) override;
    // };

    // DECLARE_RHI_COMMAND(RhiCmd_DispatchIndirect)
    // {
    //     const RhiBuffer* mBuffer;
    //     u32              mOffset;

    //     RhiCmd_DispatchIndirect(const RhiBuffer* buffer, u32 offset) : mBuffer(buffer), mOffset(offset) {}
    //     void Execute(const RhiCommandListBase* cmd) override;
    // };

    // DECLARE_RHI_COMMAND(RhiCmd_DrawMeshTasks)
    // {
    //     u32 mGroupCountX;
    //     u32 mGroupCountY;
    //     u32 mGroupCountZ;

    //     RhiCmd_DrawMeshTasks(u32 groupCountX, u32 groupCountY, u32 groupCountZ)
    //         : mGroupCountX(groupCountX), mGroupCountY(groupCountY), mGroupCountZ(groupCountZ)
    //     {
    //     }
    //     void Execute(const RhiCommandListBase* cmd) override;
    // };
    // DECLARE_RHI_COMMAND(RhiCmd_DrawMeshTasksIndirect)
    // {
    //     const RhiBuffer* mBuffer;
    //     u32              mOffset;
    //     u32              mDrawCount;
    //     u32              mStride;

    //     RhiCmd_DrawMeshTasksIndirect(const RhiBuffer* buffer, u32 offset, u32 drawCount, u32 stride)
    //         : mBuffer(buffer), mOffset(offset), mDrawCount(drawCount), mStride(stride)
    //     {
    //     }
    //     void Execute(const RhiCommandListBase* cmd) override;
    // };
    // DECLARE_RHI_COMMAND(RhiCmd_DrawPrimitives)
    // {
    //     u32 mPrimCount;
    //     u32 mInstanceCount;
    //     u32 mFirstVertex;

    //     RhiCmd_DrawPrimitives(u32 primitiveCounts, u32 instanceCount, u32 firstVertex)
    //         : mPrimCount(primitiveCounts), mInstanceCount(instanceCount), mFirstVertex(firstVertex)
    //     {
    //     }
    //     void Execute(const RhiCommandListBase* cmd) override;
    // };

    // DECLARE_RHI_COMMAND(RhiCmd_DrawPrimitivesIndexed)
    // {
    //     RhiBuffer* mIndexBuffer;
    //     u32        mBaseVertexIndex;
    //     u32        mFirstInstance;
    //     u32        mStartIndex;
    //     u32        mNumIndices;
    //     u32        mNumInstances;

    //     RhiCmd_DrawPrimitivesIndexed(RhiBuffer * indexBuffer, u32 baseVertexIndex, u32 firstInstance, u32 startIndex,
    //         u32 numIndices, u32 numInstances)
    //         : mIndexBuffer(indexBuffer)
    //         , mBaseVertexIndex(baseVertexIndex)
    //         , mFirstInstance(firstInstance)
    //         , mStartIndex(startIndex)
    //         , mNumIndices(numIndices)
    //         , mNumInstances(numInstances)
    //     {
    //     }

    //     void Execute(const RhiCommandListBase* cmd) override;
    // };

    // // Graphics Pass Specific Commands
    // DECLARE_RHI_COMMAND(RhiCmd_SetViewport)
    // {
    //     Vec<RhiViewport> mViewports;

    //     RhiCmd_SetViewport(const Vec<RhiViewport>& viewports) : mViewports(viewports) {}
    //     void Execute(const RhiCommandListBase* cmd) override;
    // };

    // DECLARE_RHI_COMMAND(RhiCmd_SetScissor)
    // {
    //     Vec<RhiScissor> mScissors;

    //     RhiCmd_SetScissor(const Vec<RhiScissor>& scissors) : mScissors(scissors) {}
    //     void Execute(const RhiCommandListBase* cmd) override;
    // };
} // namespace Ifrit::RHI