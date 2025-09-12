#include "ifrit/vkrhi2/adapter/PipelineState.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/Shader.h"
#include "ifrit/vkrhi2/adapter/DescriptorHeap.h"
#include "ifrit/vkrhi2/util/Log.h"
namespace Ifrit::RHI::VulkanRHI2
{
    // ===== Compute Pipeline State =====
    struct VA_ComputePipelineStateInternal
    {
        VkPipeline       mPipeline       = VK_NULL_HANDLE;
        VkPipelineLayout mPipelineLayout = VK_NULL_HANDLE;
    };
    IFRIT_VKRHI2_API VA_ComputePipelineState::VA_ComputePipelineState(
        const RHI::RhiComputePipelineStateDesc& desc, VA_Device* device)
        : RhiComputePipeline(desc)
    {
        mData    = new VA_ComputePipelineStateInternal();
        mContext = device;

        auto                       setLayout = device->GetBindlessDescriptorHeap()->GetDescriptorSetLayout();

        VkPipelineLayoutCreateInfo layoutCI = {};
        layoutCI.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount             = 1;
        layoutCI.pSetLayouts                = &setLayout;
        layoutCI.pushConstantRangeCount     = desc.mComputeShader.mVariant->GetRefl_PushConstantSize() > 0 ? 1 : 0;

        VkPushConstantRange pushConstantRange = {};
        if (layoutCI.pushConstantRangeCount > 0)
        {
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset     = 0;
            pushConstantRange.size       = desc.mComputeShader.mVariant->GetRefl_PushConstantSize();
            layoutCI.pPushConstantRanges = &pushConstantRange;
        }
        VA_AssertResult(vkCreatePipelineLayout(device->GetVulkanDevice(), &layoutCI, nullptr, &mData->mPipelineLayout),
            "Failed to create pipeline layout");

        VkComputePipelineCreateInfo pipelineCI = {};
        pipelineCI.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCI.layout                      = mData->mPipelineLayout;
        pipelineCI.stage = static_cast<VA_ShaderVariant*>(desc.mComputeShader.mVariant)->GetShaderStageInfo();
        pipelineCI.flags = 0;
        pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
        pipelineCI.basePipelineIndex  = 0;
        VA_AssertResult(vkCreateComputePipelines(
                            device->GetVulkanDevice(), VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &mData->mPipeline),
            "Failed to create compute pipeline");
    }
    IFRIT_VKRHI2_API VA_ComputePipelineState::~VA_ComputePipelineState()
    {
        IF_LOG_DEBUG("VA_ComputePipelineState", "Destroying compute pipeline state");
        if (mData->mPipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(static_cast<VA_Device*>(mContext)->GetVulkanDevice(), mData->mPipeline, nullptr);
            mData->mPipeline = VK_NULL_HANDLE;
        }
        if (mData->mPipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(
                static_cast<VA_Device*>(mContext)->GetVulkanDevice(), mData->mPipelineLayout, nullptr);
            mData->mPipelineLayout = VK_NULL_HANDLE;
        }
        delete mData;
        mData = nullptr;
    }
    VkPipeline       VA_ComputePipelineState::GetVulkanPipeline() const { return mData->mPipeline; }
    VkPipelineLayout VA_ComputePipelineState::GetVulkanPipelineLayout() const { return mData->mPipelineLayout; }

    // ===== Graphics Pipeline State =====
    struct VA_GraphicsPipelineStateInternal
    {
        VkPipeline       mPipeline       = VK_NULL_HANDLE;
        VkPipelineLayout mPipelineLayout = VK_NULL_HANDLE;
    };
    IFRIT_VKRHI2_API VA_GraphicsPipelineState::VA_GraphicsPipelineState(
        const RHI::RhiGraphicsPipelineStateDesc& desc, VA_Device* device)
        : RhiGraphicsPipeline(desc)
    {
        mData    = new VA_GraphicsPipelineStateInternal();
        mContext = device;

        RhiShaderVariant* const* pShaderVariants[] = { &desc.mAmplificationShader.mVariant, &desc.mMeshShader.mVariant,
            &desc.mVertexShader.mVariant, &desc.mGeometryShader.mVariant, &desc.mPixelShader.mVariant };

        static VkDynamicState    dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
               VK_DYNAMIC_STATE_STENCIL_REFERENCE };

        // layout first
        auto                     setLayout  = device->GetBindlessDescriptorHeap()->GetDescriptorSetLayout();
        VkPipelineLayoutCreateInfo layoutCI = {};
        layoutCI.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount             = 1;
        layoutCI.pSetLayouts                = &setLayout;

        u32 pushConstantSize = ~0;
        for (u32 i = 0; i < 5; ++i)
        {
            if (pShaderVariants[i] && *pShaderVariants[i])
            {
                auto shaderPushConstantSize =
                    static_cast<VA_ShaderVariant*>(*pShaderVariants[i])->GetRefl_PushConstantSize();
                if (pushConstantSize == ~0)
                    pushConstantSize = shaderPushConstantSize;
                else if (pushConstantSize != shaderPushConstantSize)
                {
                    IF_LOG_ERROR("VA_PipelineState", "Mismatched push constant size between shader stages");
                }
            }
        }
        VkPushConstantRange pushConstantRange = {};
        if (pushConstantSize != ~0 && pushConstantSize > 0)
        {
            layoutCI.pushConstantRangeCount = 1;
            pushConstantRange.stageFlags    = VK_SHADER_STAGE_ALL_GRAPHICS;
            pushConstantRange.offset        = 0;
            pushConstantRange.size          = pushConstantSize;
            layoutCI.pPushConstantRanges    = &pushConstantRange;
        }
        else
        {
            pushConstantSize                = 0;
            layoutCI.pushConstantRangeCount = 0;
        }

        VA_AssertResult(vkCreatePipelineLayout(device->GetVulkanDevice(), &layoutCI, nullptr, &mData->mPipelineLayout),
            "Failed to create pipeline layout");

        // Pipeline next
        Vec<VkPipelineShaderStageCreateInfo> shaderStages;
        for (u32 i = 0; i < 5; ++i)
        {
            if (pShaderVariants[i] && *pShaderVariants[i])
            {
                shaderStages.push_back(static_cast<VA_ShaderVariant*>(*pShaderVariants[i])->GetShaderStageInfo());
            }
        }

        // Pipeline State Create Info
        VkGraphicsPipelineCreateInfo pipelineCI = {};
        pipelineCI.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCI.layout                       = mData->mPipelineLayout;
        pipelineCI.renderPass                   = nullptr;
        pipelineCI.subpass                      = 0;
        pipelineCI.pNext                        = nullptr;
        pipelineCI.stageCount                   = SizeCast<u32>(shaderStages.size());
        pipelineCI.pStages                      = shaderStages.data();
        pipelineCI.basePipelineHandle           = VK_NULL_HANDLE;
        pipelineCI.basePipelineIndex            = 0;
        pipelineCI.flags                        = 0;

        pipelineCI.pStages = shaderStages.data();

        // Vertex Input State
        VkPipelineVertexInputStateCreateInfo vertexInputCI = {};
        vertexInputCI.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputCI.vertexBindingDescriptionCount        = 0;
        vertexInputCI.pVertexBindingDescriptions           = nullptr;
        vertexInputCI.vertexAttributeDescriptionCount      = 0;
        vertexInputCI.pVertexAttributeDescriptions         = nullptr;
        vertexInputCI.flags                                = 0;
        vertexInputCI.pNext                                = nullptr;
        pipelineCI.pVertexInputState                       = &vertexInputCI;

        // Input Assembly State
        auto fnTranslateTopology = [](ERhiRasterizerTopology topology) -> VkPrimitiveTopology {
            switch (topology)
            {
                case ERhiRasterizerTopology::TriangleList:
                    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                case ERhiRasterizerTopology::Line:
                    return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
                case ERhiRasterizerTopology::Point:
                    return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
                default:
                    IF_LOG_CRITICAL("VA_Pipeline", "Unknown topology");
                    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            }
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyCI = {};
        inputAssemblyCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyCI.topology               = fnTranslateTopology(desc.mRasterizerTopology);
        inputAssemblyCI.primitiveRestartEnable = VK_FALSE;
        inputAssemblyCI.pNext                  = nullptr;
        inputAssemblyCI.flags                  = 0;
        pipelineCI.pInputAssemblyState         = &inputAssemblyCI;

        // Tessellation State
        pipelineCI.pTessellationState = nullptr;

        // Viewport State (dyanamic state)
        VkPipelineViewportStateCreateInfo viewportCI = {};
        viewportCI.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportCI.viewportCount                     = 1;
        viewportCI.scissorCount                      = 1;
        pipelineCI.pViewportState                    = &viewportCI;

        // Rasterization State
        auto fnTranslateCullingMode = [](ERhiCullMode cullMode) -> VkCullModeFlags {
            switch (cullMode)
            {
                case ERhiCullMode::None:
                    return VK_CULL_MODE_NONE;
                case ERhiCullMode::Front:
                    return VK_CULL_MODE_FRONT_BIT;
                case ERhiCullMode::Back:
                    return VK_CULL_MODE_BACK_BIT;
                default:
                    IF_LOG_CRITICAL("VA_Pipeline", "Unknown culling mode");
                    return VK_CULL_MODE_BACK_BIT;
            }
        };
        auto fnTranslateFrontFace = [](ERhiFrontFace frontFace) -> VkFrontFace {
            switch (frontFace)
            {
                case ERhiFrontFace::Clockwise:
                    return VK_FRONT_FACE_CLOCKWISE;
                case ERhiFrontFace::CounterClockwise:
                    return VK_FRONT_FACE_COUNTER_CLOCKWISE;
                default:
                    IF_LOG_CRITICAL("VA_Pipeline", "Unknown front face");
                    return VK_FRONT_FACE_CLOCKWISE;
            }
        };

        VkPipelineRasterizationStateCreateInfo rasterizationCI = {};
        rasterizationCI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationCI.depthClampEnable        = VK_FALSE;
        rasterizationCI.rasterizerDiscardEnable = VK_FALSE;
        rasterizationCI.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizationCI.lineWidth               = 1.0f;
        rasterizationCI.cullMode                = fnTranslateCullingMode(desc.mCullMode);
        rasterizationCI.frontFace               = fnTranslateFrontFace(desc.mFrontFace);
        rasterizationCI.depthBiasEnable         = VK_FALSE;
        rasterizationCI.pNext                   = nullptr;
        rasterizationCI.flags                   = 0;
        pipelineCI.pRasterizationState          = &rasterizationCI;

        // Multisampling
        auto checkSampleCountSupported = [device](
                                             u32 sampleCount, VkSampleCountFlagBits flag) -> VkSampleCountFlagBits {
            auto res = (device->GetProperties().mRTSamplesSupported & sampleCount) != 0;
            if (!res)
            {
                IF_LOG_WARNING("VA_Pipeline", "Requested MSAA sample count {} is not supported", sampleCount);
                return VK_SAMPLE_COUNT_1_BIT;
            }
            return flag;
        };

        VkPipelineMultisampleStateCreateInfo multisampleCI = {};
        multisampleCI.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleCI.sampleShadingEnable                  = VK_FALSE;
        switch (desc.mMsaaSamples)
        {
            case 1:
                multisampleCI.rasterizationSamples = checkSampleCountSupported(1, VK_SAMPLE_COUNT_1_BIT);
                break;
            case 2:
                multisampleCI.rasterizationSamples = checkSampleCountSupported(2, VK_SAMPLE_COUNT_2_BIT);
                break;
            case 4:
                multisampleCI.rasterizationSamples = checkSampleCountSupported(4, VK_SAMPLE_COUNT_4_BIT);
                break;
            case 8:
                multisampleCI.rasterizationSamples = checkSampleCountSupported(8, VK_SAMPLE_COUNT_8_BIT);
                break;
            case 16:
                multisampleCI.rasterizationSamples = checkSampleCountSupported(16, VK_SAMPLE_COUNT_16_BIT);
                break;
            case 32:
                multisampleCI.rasterizationSamples = checkSampleCountSupported(32, VK_SAMPLE_COUNT_32_BIT);
                break;
            case 64:
                multisampleCI.rasterizationSamples = checkSampleCountSupported(64, VK_SAMPLE_COUNT_64_BIT);
                break;
            default:
                IF_LOG_WARNING("VA_Pipeline", "Unsupported MSAA sample count {}, using 1 instead", desc.mMsaaSamples);
                multisampleCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
                break;
        }
        multisampleCI.pNext          = nullptr;
        multisampleCI.flags          = 0;
        pipelineCI.pMultisampleState = &multisampleCI;

        // Depth Stencil State
        auto fnTranslateCompareOp = [](ERhiCompareOp func) -> VkCompareOp {
            switch (func)
            {
                case ERhiCompareOp::Never:
                    return VK_COMPARE_OP_NEVER;
                case ERhiCompareOp::Less:
                    return VK_COMPARE_OP_LESS;
                case ERhiCompareOp::Equal:
                    return VK_COMPARE_OP_EQUAL;
                case ERhiCompareOp::LessOrEqual:
                    return VK_COMPARE_OP_LESS_OR_EQUAL;
                case ERhiCompareOp::Greater:
                    return VK_COMPARE_OP_GREATER;
                case ERhiCompareOp::NotEqual:
                    return VK_COMPARE_OP_NOT_EQUAL;
                case ERhiCompareOp::GreaterOrEqual:
                    return VK_COMPARE_OP_GREATER_OR_EQUAL;
                case ERhiCompareOp::Always:
                    return VK_COMPARE_OP_ALWAYS;
                default:
                    IF_LOG_CRITICAL("VA_Pipeline", "Unknown compare op");
                    return VK_COMPARE_OP_ALWAYS;
            }
        };

        auto fnTranslateStencilOp = [](ERhiStencilOp op) -> VkStencilOp {
            switch (op)
            {
                case ERhiStencilOp::Keep:
                    return VK_STENCIL_OP_KEEP;
                case ERhiStencilOp::Zero:
                    return VK_STENCIL_OP_ZERO;
                case ERhiStencilOp::Replace:
                    return VK_STENCIL_OP_REPLACE;
                case ERhiStencilOp::IncrementAndClamp:
                    return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
                case ERhiStencilOp::DecrementAndClamp:
                    return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
                case ERhiStencilOp::Invert:
                    return VK_STENCIL_OP_INVERT;
                case ERhiStencilOp::IncrementAndWrap:
                    return VK_STENCIL_OP_INCREMENT_AND_WRAP;
                case ERhiStencilOp::DecrementAndWrap:
                    return VK_STENCIL_OP_DECREMENT_AND_WRAP;
                default:
                    IF_LOG_CRITICAL("VA_Pipeline", "Unknown stencil op");
                    return VK_STENCIL_OP_KEEP;
            }
        };

        // Note: following dx12, stencil ref is set to dynamic state
        VkPipelineDepthStencilStateCreateInfo depthStencilCI = {};
        depthStencilCI.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilCI.depthTestEnable       = desc.mDepthTestEnable ? VK_TRUE : VK_FALSE;
        depthStencilCI.depthWriteEnable      = desc.mDepthWriteEnable ? TRUE : VK_FALSE;
        depthStencilCI.depthCompareOp        = fnTranslateCompareOp(desc.mDepthCompareOp);
        depthStencilCI.depthBoundsTestEnable = VK_FALSE;
        depthStencilCI.stencilTestEnable     = desc.mStencilTestEnable ? VK_TRUE : VK_FALSE;

        VkStencilOpState stencilOpState = {};
        stencilOpState.failOp           = fnTranslateStencilOp(desc.mStencilFailOp);
        stencilOpState.passOp           = fnTranslateStencilOp(desc.mStencilPassOp);
        stencilOpState.depthFailOp      = fnTranslateStencilOp(desc.mStencilDepthFailOp);
        stencilOpState.compareOp        = fnTranslateCompareOp(desc.mStencilCompareOp);
        stencilOpState.compareMask      = 0xFFFFFFFF;
        stencilOpState.writeMask        = 0xFFFFFFFF;
        stencilOpState.reference        = 0; // dynamic state

        depthStencilCI.front          = stencilOpState;
        depthStencilCI.back           = stencilOpState;
        depthStencilCI.minDepthBounds = 0.0f;
        depthStencilCI.maxDepthBounds = 1.0f;
        depthStencilCI.pNext          = nullptr;
        depthStencilCI.flags          = 0;
        pipelineCI.pDepthStencilState = &depthStencilCI;

        // Dynamic State
        VkPipelineDynamicStateCreateInfo dynamicStateCI = {};
        dynamicStateCI.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateCI.pNext                            = nullptr;
        dynamicStateCI.dynamicStateCount                = 3;
        dynamicStateCI.pDynamicStates                   = dynamicStates;
        dynamicStateCI.flags                            = 0;
        pipelineCI.pDynamicState                        = &dynamicStateCI;

        // Color Blend State
        auto translateBlendOp = [](ERhiBlendOp op) -> VkBlendOp {
            switch (op)
            {
                case ERhiBlendOp::Add:
                    return VK_BLEND_OP_ADD;
                case ERhiBlendOp::Subtract:
                    return VK_BLEND_OP_SUBTRACT;
                case ERhiBlendOp::Min:
                    return VK_BLEND_OP_MIN;
                case ERhiBlendOp::Max:
                    return VK_BLEND_OP_MAX;
                default:
                    IF_LOG_CRITICAL("VA_Pipeline", "Unknown blend op");
                    return VK_BLEND_OP_ADD;
            }
        };
        auto translateBlendFactor = [](ERhiBlendFactor mode) -> VkBlendFactor {
            switch (mode)
            {
                case ERhiBlendFactor::Zero:
                    return VK_BLEND_FACTOR_ZERO;
                case ERhiBlendFactor::One:
                    return VK_BLEND_FACTOR_ONE;
                case ERhiBlendFactor::SrcColor:
                    return VK_BLEND_FACTOR_SRC_COLOR;
                case ERhiBlendFactor::OneMinusSrcColor:
                    return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                case ERhiBlendFactor::DstColor:
                    return VK_BLEND_FACTOR_DST_COLOR;
                case ERhiBlendFactor::OneMinusDstColor:
                    return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
                case ERhiBlendFactor::SrcAlpha:
                    return VK_BLEND_FACTOR_SRC_ALPHA;
                case ERhiBlendFactor::OneMinusSrcAlpha:
                    return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                case ERhiBlendFactor::DstAlpha:
                    return VK_BLEND_FACTOR_DST_ALPHA;
                case ERhiBlendFactor::OneMinusDstAlpha:
                    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
                case ERhiBlendFactor::ConstantColor:
                    return VK_BLEND_FACTOR_CONSTANT_COLOR;
            }
            IF_LOG_CRITICAL("VA_Pipeline", "Unknown blend factor");
            return VK_BLEND_FACTOR_ZERO;
        };

        VkPipelineColorBlendStateCreateInfo colorBlendCI = {};
        colorBlendCI.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendCI.logicOpEnable                       = VK_FALSE;
        colorBlendCI.logicOp                             = VK_LOGIC_OP_COPY;
        colorBlendCI.attachmentCount                     = SizeCast<u32>(desc.mBlendState.mRTBlend.size());
        Vec<VkPipelineColorBlendAttachmentState> blendAttachments;
        blendAttachments.resize(desc.mBlendState.mRTBlend.size());
        for (u32 i = 0; i < desc.mBlendState.mRTBlend.size(); ++i)
        {
            const auto& rtBlend = desc.mBlendState.mRTBlend[i];
            auto&       att     = blendAttachments[i];
            att.colorWriteMask  = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                | VK_COLOR_COMPONENT_A_BIT;
            att.blendEnable = rtBlend.mEnableBlend ? VK_TRUE : VK_FALSE;
            if (rtBlend.mEnableBlend)
            {
                att.srcColorBlendFactor = translateBlendFactor(rtBlend.mSrcBlend);
                att.dstColorBlendFactor = translateBlendFactor(rtBlend.mDstBlend);
                att.colorBlendOp        = translateBlendOp(rtBlend.mBlendOp);
                att.srcAlphaBlendFactor = translateBlendFactor(rtBlend.mSrcAlphaBlend);
                att.dstAlphaBlendFactor = translateBlendFactor(rtBlend.mDstAlphaBlend);
                att.alphaBlendOp        = translateBlendOp(rtBlend.mAlphaBlendOp);
            }
        }
        colorBlendCI.pAttachments      = blendAttachments.data();
        colorBlendCI.blendConstants[0] = 1.0f;
        colorBlendCI.blendConstants[1] = 1.0f;
        colorBlendCI.blendConstants[2] = 1.0f;
        colorBlendCI.blendConstants[3] = 1.0f;
        colorBlendCI.pNext             = nullptr;
        colorBlendCI.flags             = 0;
        pipelineCI.pColorBlendState    = &colorBlendCI;

        VA_AssertResult(vkCreateGraphicsPipelines(
                            device->GetVulkanDevice(), VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &mData->mPipeline),
            "Failed to create graphics pipeline");
    }
    IFRIT_VKRHI2_API VA_GraphicsPipelineState::~VA_GraphicsPipelineState()
    {
        if (mData->mPipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(static_cast<VA_Device*>(mContext)->GetVulkanDevice(), mData->mPipeline, nullptr);
            mData->mPipeline = VK_NULL_HANDLE;
        }
        if (mData->mPipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(
                static_cast<VA_Device*>(mContext)->GetVulkanDevice(), mData->mPipelineLayout, nullptr);
            mData->mPipelineLayout = VK_NULL_HANDLE;
        }
        delete mData;
        mData = nullptr;
    }
    VkPipeline       VA_GraphicsPipelineState::GetVulkanPipeline() const { return mData->mPipeline; }
    VkPipelineLayout VA_GraphicsPipelineState::GetVulkanPipelineLayout() const { return mData->mPipelineLayout; }

    // ===== Pipeline State Cache Registry =====
    struct VA_PipelineStateCacheRegistryInternal : public NonCopyable
    {
        HashMap<u64, Owner<VA_ComputePipelineState>>  mComputePipelines;
        HashMap<u64, Owner<VA_GraphicsPipelineState>> mGraphicsPipelines;
        VA_Device*                                    mDevice = nullptr;
        Mutex                                         mMutex;
    };
    IFRIT_VKRHI2_API VA_PipelineStateCacheRegistry::VA_PipelineStateCacheRegistry(VA_Device* device)
    {
        mInternal          = new VA_PipelineStateCacheRegistryInternal();
        mInternal->mDevice = device;
    }
    IFRIT_VKRHI2_API VA_PipelineStateCacheRegistry::~VA_PipelineStateCacheRegistry()
    {
        IF_LOG_INFO("VA_PipelineStateCacheRegistry", "Destroying pipeline state cache registry");
        delete mInternal;
        mInternal = nullptr;
    }
    IFRIT_VKRHI2_API VA_ComputePipelineState* VA_PipelineStateCacheRegistry::GetComputePipeline(
        const RHI::RhiComputePipelineStateDesc& desc)
    {
        ScopedLock lock(mInternal->mMutex);
        u64        hash = desc.Hash();
        auto       it   = mInternal->mComputePipelines.find(hash);
        if (it != mInternal->mComputePipelines.end())
        {
            return it->second.get();
        }
        CreateComputePipeline(desc);
        it = mInternal->mComputePipelines.find(hash);
        if (it != mInternal->mComputePipelines.end())
        {
            return it->second.get();
        }
        return nullptr;
    }
    IFRIT_VKRHI2_API VA_GraphicsPipelineState* VA_PipelineStateCacheRegistry::GetGraphicsPipeline(
        const RHI::RhiGraphicsPipelineStateDesc& desc)
    {
        ScopedLock lock(mInternal->mMutex);
        u64        hash = desc.Hash();
        auto       it   = mInternal->mGraphicsPipelines.find(hash);
        if (it != mInternal->mGraphicsPipelines.end())
        {
            return it->second.get();
        }
        CreateGraphicsPipeline(desc);
        it = mInternal->mGraphicsPipelines.find(hash);
        if (it != mInternal->mGraphicsPipelines.end())
        {
            return it->second.get();
        }
        return nullptr;
    }
    IFRIT_VKRHI2_API void VA_PipelineStateCacheRegistry::CreateComputePipeline(
        const RHI::RhiComputePipelineStateDesc& desc)
    {
        u64  hash                          = desc.Hash();
        auto pipeline                      = MakeOwner<VA_ComputePipelineState>(desc, mInternal->mDevice);
        mInternal->mComputePipelines[hash] = std::move(pipeline);
    }
    IFRIT_VKRHI2_API void VA_PipelineStateCacheRegistry::CreateGraphicsPipeline(
        const RHI::RhiGraphicsPipelineStateDesc& desc)
    {
        u64  hash                           = desc.Hash();
        auto pipeline                       = MakeOwner<VA_GraphicsPipelineState>(desc, mInternal->mDevice);
        mInternal->mGraphicsPipelines[hash] = std::move(pipeline);
    }

} // namespace Ifrit::RHI::VulkanRHI2