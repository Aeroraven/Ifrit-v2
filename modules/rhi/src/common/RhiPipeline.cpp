#include "ifrit/rhi/common/RhiPipeline.h"

namespace Ifrit::RHI
{
    // ===== Pipeline State =====

    IFRIT_RHI_API bool RhiComputePipelineStateDesc::operator==(const RhiComputePipelineStateDesc& other) const
    {
        return (mDescriptorLayout == other.mDescriptorLayout) && (mComputeShader == other.mComputeShader);
    }

    IFRIT_RHI_API u64 RhiComputePipelineStateDesc::Hash() const
    {
        u64 seed = mDescriptorLayout.Hash();
        seed ^= mComputeShader.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    IFRIT_RHI_API bool RhiGraphicsPipelineStateDesc::operator==(const RhiGraphicsPipelineStateDesc& other) const
    {
        if (!(mDescriptorLayout == other.mDescriptorLayout))
            return false;
        if (mVertexGeneration != other.mVertexGeneration)
            return false;
        if (!(mFrameBufferFormat == other.mFrameBufferFormat))
            return false;
        if (!(mAmplificationShader == other.mAmplificationShader))
            return false;
        if (!(mMeshShader == other.mMeshShader))
            return false;
        if (!(mVertexShader == other.mVertexShader))
            return false;
        if (!(mPixelShader == other.mPixelShader))
            return false;
        if (!(mGeometryShader == other.mGeometryShader))
            return false;
        if (mRasterizerTopology != other.mRasterizerTopology)
            return false;
        if (mDepthCompareOp != other.mDepthCompareOp)
            return false;
        if (mDepthTestEnable != other.mDepthTestEnable)
            return false;
        if (mMsaaSamples != other.mMsaaSamples)
            return false;
        if (mStencilTestEnable != other.mStencilTestEnable)
            return false;
        if (mStencilCompareOp != other.mStencilCompareOp)
            return false;
        if (mStencilFailOp != other.mStencilFailOp)
            return false;
        if (mStencilDepthFailOp != other.mStencilDepthFailOp)
            return false;
        if (mStencilPassOp != other.mStencilPassOp)
            return false;
        if (mDepthWriteEnable != other.mDepthWriteEnable)
            return false;
        if (mCullMode != other.mCullMode)
            return false;
        if (mFrontFace != other.mFrontFace)
            return false;
        if (!(mBlendState == other.mBlendState))
            return false;

        return true;
    }

    IFRIT_RHI_API u64 RhiGraphicsPipelineStateDesc::Hash() const
    {
        u64 seed = mDescriptorLayout.Hash();
        seed ^= (u32)mVertexGeneration + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= mFrameBufferFormat.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= mAmplificationShader.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= mMeshShader.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= mVertexShader.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= mPixelShader.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= mGeometryShader.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mRasterizerTopology + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mDepthCompareOp + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mDepthTestEnable + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mMsaaSamples + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mStencilTestEnable + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mStencilCompareOp + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mStencilFailOp + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mStencilDepthFailOp + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mStencilPassOp + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mDepthWriteEnable + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mCullMode + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (u32)mFrontFace + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= mBlendState.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        return seed;
    }
} // namespace Ifrit::RHI
