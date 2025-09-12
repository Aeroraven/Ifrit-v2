#pragma once
#include "ifrit/rhi/common/RhiApi.h"
#include "RhiBaseTypes.h"
#include "RhiResource.h"
#include "RhiShaderResource.h"
#include "RhiResourceView.h"
#include "ifrit/core/algo/Hash.h"

namespace Ifrit::RHI
{
    // ===== Pipeline Descriptor =====
    struct RhiPipelineDescriptorDesc
    {
        ERhiDescriptorHeapType mType     = ERhiDescriptorHeapType::Empty;
        u32                    mCount    = 0;
        bool                   mBindless = false;

        bool                   operator==(const RhiPipelineDescriptorDesc& other) const
        {
            if (mType == other.mType)
            {
                if (mBindless == other.mBindless)
                {
                    return mCount == other.mCount;
                }
                return false;
            }
            return false;
        }
        u64 Hash() const { return HashCombine(mType, mCount, mBindless); }
    };

    struct RhiPipelineDescriptorBinding
    {
        Vec<RhiPipelineDescriptorDesc> mDesc;
        bool                           operator==(const RhiPipelineDescriptorBinding& other) const
        {
            if (mDesc.size() != other.mDesc.size())
                return false;
            for (u32 i = 0; i < mDesc.size(); i++)
            {
                if (!(mDesc[i] == other.mDesc[i]))
                    return false;
            }
            return true;
        }
        u64 Hash() const
        {
            u64 seed = 0;
            for (const auto& desc : mDesc)
            {
                seed ^= desc.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    struct RhiPipelineDescriptorLayout
    {
        Vec<RhiPipelineDescriptorBinding> mSetLayouts;
        bool                              operator==(const RhiPipelineDescriptorLayout& other) const
        {
            if (mSetLayouts.size() != other.mSetLayouts.size())
                return false;
            for (u32 i = 0; i < mSetLayouts.size(); i++)
            {
                if (!(mSetLayouts[i] == other.mSetLayouts[i]))
                    return false;
            }
            return true;
        }
        u64 Hash() const
        {
            u64 seed = 0;
            for (const auto& set : mSetLayouts)
            {
                seed ^= set.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    // ===== Pipeline State =====

    enum class ERhiGraphicsPipelineVertexGeneration
    {
        VertexShader,
        MeshShader
    };

    enum class ERhiPipelineBindpoint
    {
        Compute,
        Graphics,
        RayTracing
    };

    struct RhiComputePipelineStateDesc
    {
        RhiPipelineDescriptorLayout mDescriptorLayout;
        RhiShaderVariantDesc        mComputeShader;

        IFRIT_RHI_API bool          operator==(const RhiComputePipelineStateDesc& other) const;
        IFRIT_RHI_API u64           Hash() const;
    };

    struct RhiSingleRTBlendDesc
    {
        bool            mEnableBlend   = false;
        ERhiBlendOp     mBlendOp       = ERhiBlendOp::Add;
        ERhiBlendOp     mAlphaBlendOp  = ERhiBlendOp::Add;
        ERhiBlendFactor mSrcBlend      = ERhiBlendFactor::One;
        ERhiBlendFactor mDstBlend      = ERhiBlendFactor::Zero;
        ERhiBlendFactor mSrcAlphaBlend = ERhiBlendFactor::One;
        ERhiBlendFactor mDstAlphaBlend = ERhiBlendFactor::Zero;

        u64             Hash() const
        {
            u64 seed = static_cast<u32>(mEnableBlend) | (static_cast<u32>(mBlendOp) << 1)
                | (static_cast<u32>(mAlphaBlendOp) << 4) | (static_cast<u32>(mSrcBlend) << 7)
                | (static_cast<u32>(mDstBlend) << 11) | (static_cast<u32>(mSrcAlphaBlend) << 15)
                | (static_cast<u32>(mDstAlphaBlend) << 19);
            return seed;
        }
        bool operator==(const RhiSingleRTBlendDesc& other) const
        {
            return (mEnableBlend == other.mEnableBlend) && (mBlendOp == other.mBlendOp)
                && (mAlphaBlendOp == other.mAlphaBlendOp) && (mSrcBlend == other.mSrcBlend)
                && (mDstBlend == other.mDstBlend) && (mSrcAlphaBlend == other.mSrcAlphaBlend)
                && (mDstAlphaBlend == other.mDstAlphaBlend);
        }
    };

    struct RhiRTBlendDesc
    {
        bool                      mEnableAlphaToCoverage = false;
        Vec<RhiSingleRTBlendDesc> mRTBlend;

        u64                       Hash() const
        {
            u64 seed = static_cast<u32>(mEnableAlphaToCoverage);
            for (const auto& rtBlend : mRTBlend)
            {
                seed ^= rtBlend.Hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
        bool operator==(const RhiRTBlendDesc& other) const
        {
            if (mEnableAlphaToCoverage != other.mEnableAlphaToCoverage)
                return false;
            if (mRTBlend.size() != other.mRTBlend.size())
                return false;
            for (u32 i = 0; i < mRTBlend.size(); i++)
            {
                if (!(mRTBlend[i] == other.mRTBlend[i]))
                    return false;
            }
            return true;
        }
    };

    struct RhiGraphicsPipelineStateDesc
    {
        RhiPipelineDescriptorLayout          mDescriptorLayout;
        ERhiGraphicsPipelineVertexGeneration mVertexGeneration = ERhiGraphicsPipelineVertexGeneration::VertexShader;
        RhiRenderTargetsFormat               mFrameBufferFormat;

        RhiShaderVariantDesc                 mAmplificationShader;
        RhiShaderVariantDesc                 mMeshShader;
        RhiShaderVariantDesc                 mVertexShader;
        RhiShaderVariantDesc                 mPixelShader;
        RhiShaderVariantDesc                 mGeometryShader;

        ERhiRasterizerTopology               mRasterizerTopology = ERhiRasterizerTopology::TriangleList;
        ERhiCompareOp                        mDepthCompareOp     = ERhiCompareOp::Less;
        ERhiCompareOp                        mStencilCompareOp   = ERhiCompareOp::Always;
        ERhiStencilOp                        mStencilFailOp      = ERhiStencilOp::Keep;
        ERhiStencilOp                        mStencilDepthFailOp = ERhiStencilOp::Keep;
        ERhiStencilOp                        mStencilPassOp      = ERhiStencilOp::Keep;
        bool                                 mDepthTestEnable    = false;
        bool                                 mDepthWriteEnable   = false;
        bool                                 mStencilTestEnable  = false;
        u32                                  mMsaaSamples        = 1;
        u32                                  mStencilRef         = 0;
        ERhiCullMode                         mCullMode           = ERhiCullMode::None;
        ERhiFrontFace                        mFrontFace          = ERhiFrontFace::CounterClockwise;

        RhiRTBlendDesc                       mBlendState;

        IFRIT_RHI_API bool                   operator==(const RhiGraphicsPipelineStateDesc& other) const;
        IFRIT_RHI_API u64                    Hash() const;
    };

    // ===== Pipeline =====

    class IFRIT_RHI_API RhiComputePipeline : public RhiDeviceResource
    {
    public:
        RhiComputePipeline(const RhiComputePipelineStateDesc& desc)
            : RhiDeviceResource(ERhiResourceType::Pipeline), mDesc(desc)
        {
        }
        virtual ~RhiComputePipeline() = default;

    private:
        RhiComputePipelineStateDesc mDesc;
    };

    class IFRIT_RHI_API RhiGraphicsPipeline : public RhiDeviceResource
    {
    public:
        RhiGraphicsPipeline(const RhiGraphicsPipelineStateDesc& desc)
            : RhiDeviceResource(ERhiResourceType::Pipeline), mDesc(desc)
        {
        }
        virtual ~RhiGraphicsPipeline() = default;

    private:
        RhiGraphicsPipelineStateDesc mDesc;
    };

    // ===== Pipeline Cache (Runtime) =====
    struct RhiPipelineCacheRegistryInternal;
    class IFRIT_RHI_API RhiPipelineCacheRegistry
    {
    public:
        RhiPipelineCacheRegistry();
        ~RhiPipelineCacheRegistry();
        RhiComputePipelineRef  GetComputePipeline(const RhiComputePipelineStateDesc& desc);
        RhiGraphicsPipelineRef GetGraphicsPipeline(const RhiGraphicsPipelineStateDesc& desc);

    private:
        RhiPipelineCacheRegistryInternal* mInternal = nullptr;
    };

} // namespace Ifrit::RHI