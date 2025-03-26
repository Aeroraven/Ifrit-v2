
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "ifrit/vkgraphics/engine/vkrenderer/RenderGraph.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/utility/Logger.h"

using namespace Ifrit::Common::Utility;
namespace Ifrit::Graphics::VulkanGraphics
{
    inline VkFormat toVkFormat(Rhi::RhiImageFormat format)
    {
        return static_cast<VkFormat>(format);
    }

    template <typename E>
    IF_CONSTEXPR typename std::underlying_type<E>::type getUnderlying(E e) noexcept
    {
        return static_cast<typename std::underlying_type<E>::type>(e);
    }
    // Class : SwapchainImageResource

    IFRIT_APIDECL VkFormat SwapchainImageResource::GetFormat() const
    {
        return m_swapchain->GetPreferredFormat();
    }

    IFRIT_APIDECL VkImage SwapchainImageResource::GetImage() const
    {
        return m_swapchain->GetCurrentImage();
    }

    IFRIT_APIDECL VkImageView SwapchainImageResource::GetImageView()
    {
        return m_swapchain->GetCurrentImageView();
    }

    IFRIT_APIDECL VkImageView SwapchainImageResource::GetImageViewMipLayer(uint32_t mip, uint32_t layer, uint32_t mipRange,
        uint32_t layerRange)
    {
        if (mipRange != 1 || layerRange != 1 || mip != 0 || layer != 0)
        {
            throw std::runtime_error("Invalid mip or layer");
        }
        return m_swapchain->GetCurrentImageView();
    }

    // Class : RegisteredResource
    IFRIT_APIDECL void RegisteredResource::recordTransition(const RenderPassResourceTransition& transition,
        uint32_t                                                                                queueFamily)
    {
        m_currentLayout      = transition.m_newLayout;
        m_currentAccess      = transition.m_dstAccess;
        m_currentQueueFamily = queueFamily;
    }

    IFRIT_APIDECL void RegisteredResource::reSetState()
    {
        m_currentLayout      = m_originalLayout;
        m_currentAccess      = m_originalAccess;
        m_currentQueueFamily = m_originalQueueFamily;
    }

    // Class : RenderGraphPass

    IFRIT_APIDECL void RenderGraphPass::addInputResource(RegisteredResource* resource,
        RenderPassResourceTransition                                         transition)
    {
        m_inputResources.push_back(resource);
        m_inputTransition.push_back(transition);
    }

    IFRIT_APIDECL void RenderGraphPass::addOutputResource(RegisteredResource* resource,
        RenderPassResourceTransition                                          transition)
    {
        m_outputResources.push_back(resource);
        m_outputTransition.push_back(transition);
    }

    IFRIT_APIDECL void RenderGraphPass::setPassDescriptorLayout_base(const std::vector<Rhi::RhiDescriptorType>& layout)
    {
        m_passDescriptorLayout = layout;
    }

    IFRIT_APIDECL void RenderGraphPass::AddUniformBuffer_base(RegisteredBufferHandle* buffer, uint32_t position)
    {
        auto numCopies = buffer->GetNumBuffers();
        m_resourceDescriptorHandle[position].resize(numCopies);
        for (int i = 0; i < static_cast<int>(numCopies); i++)
        {
            auto v                                  = m_descriptorManager->RegisterUniformBuffer(buffer->GetBuffer(i));
            m_resourceDescriptorHandle[position][i] = v;
        }
        // Add as input
        RenderPassResourceTransition transition{};
        transition.m_required = false;
        addInputResource(buffer, transition);
    }

    IFRIT_APIDECL
    void RenderGraphPass::AddStorageBuffer_base(RegisteredBufferHandle* buffer, uint32_t position,
        Rhi::RhiResourceAccessType access)
    {
        auto numCopies = buffer->GetNumBuffers();
        m_resourceDescriptorHandle[position].resize(numCopies);
        for (int i = 0; i < static_cast<int>(numCopies); i++)
        {
            auto v                                  = m_descriptorManager->RegisterStorageBuffer(buffer->GetBuffer(i));
            m_resourceDescriptorHandle[position][i] = v;
        }
        // Add as input
        RenderPassResourceTransition transition{};
        transition.m_required = false;
        addInputResource(buffer, transition);
        m_ssbos.push_back(buffer);
        m_ssboAccess.push_back(access);
    }

    IFRIT_APIDECL void RenderGraphPass::AddCombinedImageSampler(RegisteredImageHandle* image,
        RegisteredSamplerHandle* sampler, uint32_t position)
    {
        auto numCopies = image->GetNumBuffers();
        m_resourceDescriptorHandle[position].resize(numCopies);
        for (int i = 0; i < static_cast<int>(numCopies); i++)
        {
            auto v                                  = m_descriptorManager->RegisterCombinedImageSampler(image->GetImage(i), sampler->GetSampler());
            m_resourceDescriptorHandle[position][i] = v;
        }
        // Add as input
        RenderPassResourceTransition transition{};
        transition.m_required = false;
        addInputResource(image, transition);
        addInputResource(sampler, transition);
    }

    IFRIT_APIDECL void RenderGraphPass::buildDescriptorParamHandle(uint32_t numMultiBuffers)
    {
        if (numMultiBuffers == 0)
        {
            numMultiBuffers = m_defaultMultibuffers;
        }
        if (m_passDescriptorLayout.size() == 0)
            return;
        for (auto& [k, v] : m_resourceDescriptorHandle)
        {
            if (v.size() != 1 && v.size() != numMultiBuffers)
            {
                vkrError("Descriptor handle size mismatch");
            }
        }
        m_descriptorBindRange.resize(numMultiBuffers);
        for (int T = 0; T < static_cast<int>(numMultiBuffers); T++)
        {
            std::vector<uint32_t> descriptorHandles;
            for (int i = 0; i < m_passDescriptorLayout.size(); i++)
            {
                auto type = m_passDescriptorLayout[i];
                if (m_resourceDescriptorHandle.count(i) == 0)
                {
                    vkrError("Descriptor handle not found for position");
                }
                auto& handleR = m_resourceDescriptorHandle[i];
                if (handleR.size() != 1 && handleR.size() != numMultiBuffers)
                {
                    vkrLog("Inconsistent double buffers");
                }
                uint32_t handle;
                // = i >= handleR.size() ? handleR.back() : handleR[T];
                if (T >= handleR.size())
                {
                    handle = handleR.back();
                }
                else
                {
                    handle = handleR[T];
                }
                descriptorHandles.push_back(handle);
            }
            m_descriptorBindRange[T] = m_descriptorManager->RegisterBindlessParameterRaw(
                reinterpret_cast<const char*>(descriptorHandles.data()),
                Ifrit::Common::Utility::SizeCast<uint32_t>(descriptorHandles.size()) * sizeof(uint32_t));
        }
    }

    IFRIT_APIDECL void
    RenderGraphPass::SetRecordFunction_base(std::function<void(Rhi::RhiRenderPassContext*)> executeFunction)
    {
        m_recordFunction = executeFunction;
    }

    IFRIT_APIDECL void RenderGraphPass::SetExecutionFunction_base(std::function<void(Rhi::RhiRenderPassContext*)> func)
    {
        m_executeFunction = func;
    }

    IFRIT_APIDECL void RenderGraphPass::execute()
    {
        if (m_executeFunction)
        {
            m_executeFunction(&m_passContext);
        }
    }

    IFRIT_APIDECL std::vector<RegisteredResource*>& RenderGraphPass::getInputResources()
    {
        return m_inputResources;
    }

    IFRIT_APIDECL std::vector<RegisteredResource*>& RenderGraphPass::getOutputResources()
    {
        return m_outputResources;
    }

    IFRIT_APIDECL void RenderGraphPass::withCommandBuffer(CommandBuffer* commandBuffer, std::function<void()> func)
    {
        m_commandBuffer = commandBuffer;
        func();
        m_commandBuffer = nullptr;
    }

    // Class : GraphicsPass
    IFRIT_APIDECL GraphicsPass::GraphicsPass(EngineContext* context, PipelineCache* pipelineCache,
        DescriptorManager* descriptorManager, RegisteredResourceMapper* mapper)
        : RenderGraphPass(context, descriptorManager, mapper), m_pipelineCache(pipelineCache) {}

    IFRIT_APIDECL void GraphicsPass::SetRenderTarGetFormat(const Rhi::RhiRenderTargetsFormat& format)
    {
        m_renderTarGetFormat = format;
    }

    IFRIT_APIDECL void GraphicsPass::SetVertexShader(Rhi::RhiShader* shader)
    {
        m_vertexShader = CheckedCast<ShaderModule>(shader);
    }

    IFRIT_APIDECL void GraphicsPass::SetPixelShader(Rhi::RhiShader* shader)
    {
        m_fragmentShader = CheckedCast<ShaderModule>(shader);
    }

    IFRIT_APIDECL void GraphicsPass::setGeometryShader(ShaderModule* shader)
    {
        m_geometryShader = shader;
    }

    IFRIT_APIDECL void GraphicsPass::setTessControlShader(ShaderModule* shader)
    {
        m_tessControlShader = shader;
    }

    IFRIT_APIDECL void GraphicsPass::setTessEvalShader(ShaderModule* shader)
    {
        m_tessEvalShader = shader;
    }

    IFRIT_APIDECL void GraphicsPass::SetTaskShader(Rhi::RhiShader* shader)
    {
        m_taskShader = CheckedCast<ShaderModule>(shader);
    }

    IFRIT_APIDECL void GraphicsPass::SetMeshShader(Rhi::RhiShader* shader)
    {
        m_meshShader = CheckedCast<ShaderModule>(shader);
    }

    IFRIT_APIDECL void GraphicsPass::record(RenderTargets* renderTarget)
    {
        m_passContext.m_cmd   = m_commandBuffer;
        m_passContext.m_frame = m_activeFrame;
        renderTarget->BeginRendering(m_commandBuffer);
        vkCmdBindPipeline(m_commandBuffer->GetCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetPipeline());
        auto exfun = m_context->GetExtensionFunction();
        auto cmd   = m_commandBuffer->GetCommandBuffer();
        if (m_meshShader == nullptr)
        {
            exfun.p_vkCmdSetVertexInputEXT(
                cmd, SizeCast<uint32_t>(m_vertexBufferDescriptor.m_bindings.size()), m_vertexBufferDescriptor.m_bindings.data(),
                SizeCast<uint32_t>(m_vertexBufferDescriptor.m_attributes.size()), m_vertexBufferDescriptor.m_attributes.data());

            std::vector<VkBuffer>     vxbuffers;
            std::vector<VkDeviceSize> offsets;
            for (int i = 0; i < m_vertexBuffers.size(); i++)
            {
                vxbuffers.push_back(m_vertexBuffers[i]->GetBuffer(m_passContext.m_frame)->GetBuffer());
                offsets.push_back(0);
            }
            if (vxbuffers.size() > 0)
            {
                vkCmdBindVertexBuffers(cmd, 0, Ifrit::Common::Utility::SizeCast<uint32_t>(m_vertexBuffers.size()),
                    vxbuffers.data(), offsets.data());
            }

            if (m_indexBuffer)
            {
                vkCmdBindIndexBuffer(cmd, m_indexBuffer->GetBuffer(m_passContext.m_frame)->GetBuffer(), 0, m_indexType);
            }
        }

        auto bindlessSet = m_descriptorManager->GetBindlessSet();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetLayout(), 0, 1, &bindlessSet, 0,
            nullptr);

        exfun.p_vkCmdSetLogicOpEnableEXT(cmd, m_logicalOpEnable);
        exfun.p_vkCmdSetLogicOpEXT(cmd, m_logicOp);
        exfun.p_vkCmdSetStencilTestEnable(cmd, m_stencilEnable);
        exfun.p_vkCmdSetStencilOp(cmd, m_stencilOp.faceMask, m_stencilOp.failOp, m_stencilOp.passOp, m_stencilOp.depthFailOp,
            m_stencilOp.compareOp);

        exfun.p_vkCmdSetDepthBoundsTestEnable(cmd, m_depthBoundTestEnable);
        // exfun.p_vkCmdSetDepthCompareOp(cmd, m_depthCompareOp);
        // exfun.p_vkCmdSetDepthTestEnable(cmd, m_depthTestEnable);
        // exfun.p_vkCmdSetDepthWriteEnable(cmd, m_depthWrite);

        vkCmdSetFrontFace(cmd, m_frontFace);
        vkCmdSetCullMode(cmd, m_cullMode);

        if (m_recordFunction)
        {
            m_recordFunction(&m_passContext);
        }

        renderTarget->EndRendering(m_commandBuffer);

        if (m_recordPostFunction)
        {
            m_recordPostFunction(&m_passContext);
        }
    }

    IFRIT_APIDECL void GraphicsPass::SetRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
    {
        m_renderArea.offset = { static_cast<int>(x), static_cast<int>(y) };
        m_renderArea.extent = { width, height };
    }

    IFRIT_APIDECL void GraphicsPass::SetDepthWrite(bool write)
    {
        m_depthWrite = write;
    }

    IFRIT_APIDECL void GraphicsPass::SetDepthTestEnable(bool enable)
    {
        m_depthTestEnable = enable;
    }
    IFRIT_APIDECL void GraphicsPass::SetDepthCompareOp(Rhi::RhiCompareOp compareOp)
    {
        VkCompareOp vkcmp;
        switch (compareOp)
        {
            case Rhi::RhiCompareOp::Never:
                vkcmp = VK_COMPARE_OP_NEVER;
                break;
            case Rhi::RhiCompareOp::Less:

                vkcmp = VK_COMPARE_OP_LESS;
                break;
            case Rhi::RhiCompareOp::Equal:
                vkcmp = VK_COMPARE_OP_EQUAL;
                break;
            case Rhi::RhiCompareOp::LessOrEqual:
                vkcmp = VK_COMPARE_OP_LESS_OR_EQUAL;
                break;
            case Rhi::RhiCompareOp::Greater:
                vkcmp = VK_COMPARE_OP_GREATER;
                break;
            case Rhi::RhiCompareOp::NotEqual:
                vkcmp = VK_COMPARE_OP_NOT_EQUAL;
                break;
            case Rhi::RhiCompareOp::GreaterOrEqual:
                vkcmp = VK_COMPARE_OP_GREATER_OR_EQUAL;
                break;
            case Rhi::RhiCompareOp::Always:
                vkcmp = VK_COMPARE_OP_ALWAYS;
                break;
            default:
                vkcmp = VK_COMPARE_OP_ALWAYS;
                break;
        }
        m_depthCompareOp = vkcmp;
    }
    IFRIT_APIDECL void GraphicsPass::SetRasterizerTopology(Rhi::RhiRasterizerTopology topology)
    {
        m_topology = topology;
    }

    IFRIT_APIDECL void GraphicsPass::setColorWrite(const std::vector<uint32_t>& write)
    {
        throw std::runtime_error("Deprecated");
    }

    IFRIT_APIDECL void GraphicsPass::setVertexInput(const VertexBufferDescriptor& descriptor,
        const std::vector<RegisteredBufferHandle*>&                               buffers)
    {
        m_vertexBufferDescriptor = descriptor;
        m_vertexBuffers          = buffers;
        if (m_vertexBuffers.size() != m_vertexBufferDescriptor.m_bindings.size())
        {
            vkrError("Num of vertex buffers and descriptor bindings are not equal");
        }
        RenderPassResourceTransition trans;
        trans.m_required = false;
        for (auto* x : buffers)
        {
            addInputResource(x, trans);
        }
    }

    IFRIT_APIDECL
    void GraphicsPass::setIndexInput(RegisteredBufferHandle* buffer, VkIndexType type)
    {
        m_indexBuffer = buffer;
        m_indexType   = type;

        RenderPassResourceTransition trans;
        trans.m_required = false;
        addInputResource(buffer, trans);
    }

    IFRIT_APIDECL void GraphicsPass::build(uint32_t numMultiBuffers)
    {
        GraphicsPipelineCreateInfo ci;
        if (m_vertexShader != nullptr)
        {
            ci.shaderModules.push_back(m_vertexShader);
            ci.geomGenType = Rhi::RhiGeometryGenerationType::Conventional;
        }
        if (m_fragmentShader != nullptr)
        {
            ci.shaderModules.push_back(m_fragmentShader);
        }
        if (m_geometryShader != nullptr)
        {
            ci.shaderModules.push_back(m_geometryShader);
        }
        if (m_tessControlShader != nullptr)
        {
            ci.shaderModules.push_back(m_tessControlShader);
        }
        if (m_tessEvalShader != nullptr)
        {
            ci.shaderModules.push_back(m_tessEvalShader);
        }
        if (m_taskShader != nullptr)
        {
            ci.shaderModules.push_back(m_taskShader);
        }
        if (m_meshShader != nullptr)
        {
            ci.shaderModules.push_back(m_meshShader);
            ci.geomGenType = Rhi::RhiGeometryGenerationType::Mesh;
            vkrAssert(m_vertexShader == nullptr, "Vertex shader should be null");
        }
        ci.viewportCount           = 1;
        ci.scissorCount            = 1;
        ci.topology                = m_topology;
        ci.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
        ci.depthAttachmentFormat   = toVkFormat(m_renderTarGetFormat.m_depthFormat);
        ci.pushConstSize           = m_pushConstSize;

        for (int i = 0; i < m_renderTarGetFormat.m_colorFormats.size(); i++)
        {
            ci.colorAttachmentFormats.push_back(toVkFormat(m_renderTarGetFormat.m_colorFormats[i]));
        }
        ci.descriptorSetLayouts.push_back(m_descriptorManager->GetBindlessLayout());
        for (uint32_t i = 0; i < m_numBindlessDescriptorSets; i++)
        {
            ci.descriptorSetLayouts.push_back(m_descriptorManager->GetParameterDescriptorSetLayout());
        }
        m_pipeline = m_pipelineCache->GetGraphicsPipeline(ci);
        setBuilt();
    }

    IFRIT_APIDECL uint32_t GraphicsPass::getRequiredQueueCapability()
    {
        return VK_QUEUE_GRAPHICS_BIT;
    }

    IFRIT_APIDECL void ComputePass::Run(const Rhi::RhiCommandList* cmd, uint32_t frameId)
    {
        if (m_passBuilt == false)
        {
            build(0);
        }
        m_commandBuffer = CheckedCast<CommandBuffer>(cmd);
        m_activeFrame   = frameId;
        execute();
        record();
    }

    // Class : ComputePass

    IFRIT_APIDECL void ComputePass::build(uint32_t numMultiBuffers)
    {
        ComputePipelineCreateInfo ci;
        ci.shaderModules = m_shaderModule;
        ci.descriptorSetLayouts.push_back(m_descriptorManager->GetBindlessLayout());
        for (uint32_t i = 0; i < m_numBindlessDescriptorSets; i++)
        {
            ci.descriptorSetLayouts.push_back(m_descriptorManager->GetParameterDescriptorSetLayout());
        }
        ci.pushConstSize = m_pushConstSize;
        m_pipeline       = m_pipelineCache->GetComputePipeline(ci);
        setBuilt();
    }

    IFRIT_APIDECL void ComputePass::SetComputeShader(Rhi::RhiShader* shader)
    {
        auto p         = CheckedCast<ShaderModule>(shader);
        m_shaderModule = p;
    }

    IFRIT_APIDECL uint32_t ComputePass::getRequiredQueueCapability()
    {
        return VK_QUEUE_COMPUTE_BIT;
    }

    IFRIT_APIDECL void ComputePass::record()
    {
        m_passContext.m_cmd         = m_commandBuffer;
        m_passContext.m_frame       = m_activeFrame;
        VkCommandBuffer cmd         = m_commandBuffer->GetCommandBuffer();
        auto            bindlessSet = m_descriptorManager->GetBindlessSet();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->GetLayout(), 0, 1, &bindlessSet, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->GetPipeline());

        if (m_recordFunction)
        {
            m_recordFunction(&m_passContext);
        }
    }

    IFRIT_APIDECL void GraphicsPass::Run(const Rhi::RhiCommandList* cmd, Rhi::RhiRenderTargets* renderTargets,
        uint32_t frameId)
    {
        if (m_passBuilt == false)
        {
            build(0);
        }
        m_commandBuffer = CheckedCast<CommandBuffer>(cmd);
        m_activeFrame   = frameId;
        execute();
        record(CheckedCast<RenderTargets>(renderTargets));
    }

    // Class: RenderGraph
    IFRIT_APIDECL
    RenderGraph::RenderGraph(EngineContext* context, DescriptorManager* descriptorManager)
    {
        m_context           = context;
        m_descriptorManager = descriptorManager;
        m_pipelineCache     = std::make_unique<PipelineCache>(context);
        m_mapper            = std::make_unique<RegisteredResourceMapper>();
    }

    IFRIT_APIDECL GraphicsPass* RenderGraph::addGraphicsPass()
    {
        auto pass = std::make_unique<GraphicsPass>(m_context, m_pipelineCache.get(), m_descriptorManager, m_mapper.get());
        auto ptr  = pass.get();
        m_passes.push_back(std::move(pass));
        return ptr;
    }

    IFRIT_APIDECL ComputePass* RenderGraph::addComputePass()
    {
        auto pass = std::make_unique<ComputePass>(m_context, m_pipelineCache.get(), m_descriptorManager, m_mapper.get());
        auto ptr  = pass.get();
        m_passes.push_back(std::move(pass));
        return ptr;
    }

    IFRIT_APIDECL RegisteredBufferHandle* RenderGraph::registerBuffer(SingleBuffer* buffer)
    {
        auto p = m_mapper->GetBufferIndex(buffer);
        auto s = dynamic_cast<RegisteredBufferHandle*>(p);
        if (s == nullptr)
        {
            vkrError("Invalid buffer");
        }
        return s;
    }

    IFRIT_APIDECL RegisteredSamplerHandle* RenderGraph::RegisterSampler(Sampler* sampler)
    {
        auto registeredSampler = std::make_unique<RegisteredSamplerHandle>(sampler);
        auto ptr               = registeredSampler.get();
        m_resources.push_back(std::move(registeredSampler));
        m_resourceMap[ptr] = SizeCast<uint32_t>(m_resources.size()) - 1;
        return ptr;
    }

    IFRIT_APIDECL RegisteredBufferHandle* RenderGraph::registerBuffer(MultiBuffer* buffer)
    {
        auto p = m_mapper->getMultiBufferIndex(buffer);
        auto s = dynamic_cast<RegisteredBufferHandle*>(p);
        if (s == nullptr)
        {
            vkrError("Invalid buffer");
        }
        return s;
    }

    IFRIT_APIDECL RegisteredImageHandle* RenderGraph::registerImage(SingleDeviceImage* image)
    {
        if (image->GetIsSwapchainImage())
        {
            auto p = dynamic_cast<SwapchainImageResource*>(image);
            if (p == nullptr)
            {
                vkrError("Invalid swapchain image");
            }
            auto registeredImage = std::make_unique<RegisteredSwapchainImage>(p);
            auto ptr             = registeredImage.get();
            m_resources.push_back(std::move(registeredImage));
            m_resourceMap[ptr] = SizeCast<uint32_t>(m_resources.size()) - 1;
            m_swapchainImageHandle.push_back(ptr);
            return ptr;
        }
        else
        {
            auto registeredImage = std::make_unique<RegisteredImageHandle>(image);
            auto ptr             = registeredImage.get();
            m_resources.push_back(std::move(registeredImage));
            m_resourceMap[ptr] = SizeCast<uint32_t>(m_resources.size()) - 1;
            return ptr;
        }
        return nullptr;
    }

    IFRIT_APIDECL void RenderGraph::build(uint32_t numMultiBuffers)
    {
        // Build all passes
        for (int i = 0; i < m_passes.size(); i++)
        {
            if (!m_passes[i]->isBuilt())
            {
                m_passes[i]->build(numMultiBuffers);
            }
        }
        int numSubgraph = 0;
        m_subgraphBelonging.resize(m_resources.size());
        for (int i = 0; i < m_resources.size(); i++)
        {
            m_subgraphBelonging[i] = UINT32_MAX;
        }

        std::unordered_set<RegisteredResource*> dependencySet;
        for (int i = 0; i < m_swapchainImageHandle.size(); i++)
        {
            dependencySet.insert(m_swapchainImageHandle[i]);
        }
        uint32_t assignedPass = 0;
        while (true)
        {
            for (int i = SizeCast<int>(m_passes.size()) - 1; i >= 0; i--)
            {
                auto pass = m_passes[i].get();
                if (m_subgraphBelonging[i] != UINT32_MAX)
                    continue;
                auto& inputResources  = pass->getInputResources();
                auto& outputResources = pass->getOutputResources();

                // If input or output contains object in dependency set
                // then add all input and output resources to dependency set
                bool  hasDependency = false;
                for (int j = 0; j < inputResources.size(); j++)
                {
                    if (dependencySet.find(inputResources[j]) != dependencySet.end())
                    {
                        hasDependency = true;
                        break;
                    }
                }
                for (int j = 0; j < outputResources.size(); j++)
                {
                    if (dependencySet.find(outputResources[j]) != dependencySet.end())
                    {
                        hasDependency = true;
                        break;
                    }
                }
                if (hasDependency)
                {
                    for (int j = 0; j < inputResources.size(); j++)
                    {
                        dependencySet.insert(inputResources[j]);
                    }
                    for (int j = 0; j < outputResources.size(); j++)
                    {
                        dependencySet.insert(outputResources[j]);
                    }
                    m_subgraphBelonging[i] = numSubgraph;

                    assignedPass++;
                    // m_subgraphs[numSubgraph].push_back(i);
                }
            }

            if (assignedPass != 0)
                numSubgraph++;
            if (assignedPass == m_passes.size())
                break;
            // Otherwise, find an unassigned pass that has no dependency
            // and add it to the dependency set
            dependencySet.clear();
            for (int i = 0; i < m_passes.size(); i++)
            {
                if (m_subgraphBelonging[i] == UINT32_MAX)
                {
                    auto  pass            = m_passes[i].get();
                    auto& inputResources  = pass->getInputResources();
                    auto& outputResources = pass->getOutputResources();
                    for (int j = 0; j < inputResources.size(); j++)
                    {
                        dependencySet.insert(inputResources[j]);
                    }
                    for (int j = 0; j < outputResources.size(); j++)
                    {
                        dependencySet.insert(outputResources[j]);
                    }
                }
            }
        }
        m_assignedQueues.resize(numSubgraph);
        // add to m_subgraphs
        m_subgraphs.resize(numSubgraph);
        m_subGraphOperatesOnSwapchain.resize(numSubgraph);
        for (int i = 0; i < m_passes.size(); i++)
        {
            m_subgraphs[m_subgraphBelonging[i]].push_back(i);
        }
    }

    IFRIT_APIDECL std::vector<std::vector<uint32_t>>& RenderGraph::getSubgraphs()
    {
        return m_subgraphs;
    }

    IFRIT_APIDECL void RenderGraph::resizeCommandBuffers(uint32_t size)
    {
        m_commandBuffers.resize(size);
    }

    IFRIT_APIDECL void RenderGraph::buildPassDescriptors(uint32_t numMultiBuffer)
    {
        for (int i = 0; i < m_passes.size(); i++)
        {
            auto p = m_passes[i].get();
            p->buildDescriptorParamHandle(numMultiBuffer);
        }
    }

    // Class : CommandExecutor
    IFRIT_APIDECL RenderGraph* CommandExecutor::CreateRenderGraph()
    {
        auto p   = std::make_unique<RenderGraph>(m_context, m_descriptorManager);
        auto ptr = p.get();
        m_renderGraph.push_back(std::move(p));
        return ptr;
    }

    IFRIT_APIDECL void CommandExecutor::setQueues(bool reqPresentQueue, int numGraphics, int numCompute, int numTransfer)
    {
        if (m_queueCollections == nullptr)
        {
            m_queueCollections = std::make_unique<QueueCollections>(m_context);
            m_queueCollections->LoadQueues();
        }
        auto                graphicsQueues = m_queueCollections->GetGraphicsQueues();
        auto                computeQueues  = m_queueCollections->GetComputeQueues();
        std::vector<Queue*> chosenQueue;
        uint32_t            numGraphicsQueues = numGraphics;
        uint32_t            numComputeQueues  = numCompute;
        if (reqPresentQueue)
        {
            VkQueue presentQueue = m_swapchain->GetPresentQueue();
            bool    found        = false;
            for (int i = 0; i < graphicsQueues.size(); i++)
            {
                if (graphicsQueues[i]->GetQueue() == presentQueue)
                {
                    chosenQueue.push_back(graphicsQueues[i]);
                    found = true;
                    numGraphicsQueues--;
                    break;
                }
            }
            if (!found)
            {
                // TODO: Wrong handling
                vkrError("Present queue not found in graphics queues");
            }
        }
        for (int i = 0; i < SizeCast<int>(numGraphicsQueues); i++)
        {
            chosenQueue.push_back(graphicsQueues[i]);
        }

        m_queuesGraphics = chosenQueue;
        chosenQueue.clear();

        // Compute queue
        for (uint32_t i = 0, j = 0; i < numComputeQueues; j++)
        {
            bool isGraphicsQueue = false;
            for (int i = 0; i < m_queuesGraphics.size(); i++)
            {
                if (m_queuesGraphics[i]->GetQueue() == computeQueues[j]->GetQueue())
                {
                    j++;
                    isGraphicsQueue = true;
                    break;
                }
            }
            if (isGraphicsQueue)
                continue;
            chosenQueue.push_back(computeQueues[j]);
            i++;
        }
        m_queuesCompute = chosenQueue;

        // Transfer queue
        chosenQueue.clear();
        auto transferQueues = m_queueCollections->GetTransferQueues();
        for (int i = 0, j = 0; i < numTransfer; j++)
        {
            bool isGraphicsQueue = false;
            bool isComputeQueue  = false;
            for (int k = 0; k < m_queuesGraphics.size(); k++)
            {
                if (m_queuesGraphics[k]->GetQueue() == transferQueues[j]->GetQueue())
                {
                    j++;
                    isGraphicsQueue = true;
                    break;
                }
            }
            for (int k = 0; k < m_queuesCompute.size(); k++)
            {
                if (m_queuesCompute[k]->GetQueue() == transferQueues[j]->GetQueue())
                {
                    j++;
                    isComputeQueue = true;
                    break;
                }
            }
            if (isGraphicsQueue || isComputeQueue)
                continue;
            chosenQueue.push_back(transferQueues[j]);
            i++;
        }
        m_queuesTransfer = chosenQueue;

        // insert all into m_queue
        m_queues.insert(m_queues.end(), m_queuesGraphics.begin(), m_queuesGraphics.end());
        m_queues.insert(m_queues.end(), m_queuesCompute.begin(), m_queuesCompute.end());
        m_queues.insert(m_queues.end(), m_queuesTransfer.begin(), m_queuesTransfer.end());
    }

    IFRIT_APIDECL void CommandExecutor::compileGraph(RenderGraph* graph, uint32_t numMultiBuffers)
    {
        m_graph = graph;
        if (m_graph->m_subgraphs.size() == 0)
        {
            graph->buildPassDescriptors(numMultiBuffers);
            m_descriptorManager->BuildBindlessParameter();
            graph->build(numMultiBuffers);
        }
        auto subgraphs = graph->getSubgraphs();
        for (int i = 0; i < m_queues.size(); i++)
        {
            auto queue       = m_queues[i];
            auto commandPool = std::make_unique<CommandPool>(m_context, queue->GetQueueFamily());
            m_graph->m_commandPools.push_back(std::move(commandPool));
        }

        std::vector<uint32_t> queueAssignedTimes(m_queues.size(), 0);
        for (int i = 0; i < subgraphs.size(); i++)
        {
            uint32_t requiredCapability = 0;
            bool     operateOnSwapchain = false;
            for (int j = 0; j < subgraphs[i].size(); j++)
            {
                auto pass = m_graph->m_passes[subgraphs[i][j]].get();
                requiredCapability |= pass->getRequiredQueueCapability();
                operateOnSwapchain |= pass->getOperatesOnSwapchain();
            }
            m_graph->m_subGraphOperatesOnSwapchain[i] = operateOnSwapchain;

            // Find a queue that has the required capability,
            // with the least number of assigned times
            uint32_t queueIndex = UINT32_MAX;
            for (int j = 0; j < m_queues.size(); j++)
            {
                if ((m_queues[j]->GetCapability() & requiredCapability) == requiredCapability)
                {
                    if (requiredCapability | VK_QUEUE_GRAPHICS_BIT)
                    {
                        queueIndex = j;
                        break;
                    }
                    if (queueAssignedTimes[j] < queueAssignedTimes[queueIndex])
                    {
                        queueIndex = j;
                    }
                }
            }
            if (queueIndex == UINT32_MAX)
            {
                vkrError("No queue found for subgraph");
            }
            queueAssignedTimes[queueIndex]++;
            m_graph->m_assignedQueues[i] = m_queues[queueIndex];
        }
        m_graph->m_comiled = true;
    }

    IFRIT_APIDECL void CommandExecutor::syncMultiBufferStateWithSwapchain()
    {
        m_resourceManager->SetActiveFrame(m_swapchain->GetCurrentImageIndex());
    }

    IFRIT_APIDECL void CommandExecutor::runRenderGraph(RenderGraph* graph)
    {
        syncMultiBufferStateWithSwapchain();
        auto numBackBuffers = m_swapchain->GetNumBackbuffers();
        auto currentFrame   = m_swapchain->GetCurrentImageIndex();
        if (!graph->m_comiled)
        {
            compileGraph(graph, numBackBuffers);
        }

        // perform 'execution' handlers attached on passes
        for (int i = 0; i < m_graph->m_passes.size(); i++)
        {
            auto pass = m_graph->m_passes[i].get();
            pass->SetActiveFrame(currentFrame);
            pass->execute();
        }

        // for each subgraph, create a command buffer
        auto subgraphs = graph->getSubgraphs();
        for (int i = 0; i < subgraphs.size(); i++)
        {
            auto                  queue = m_graph->m_assignedQueues[i];
            TimelineSemaphoreWait lastWait;
            for (int j = 0; j < subgraphs[i].size(); j++)
            {
                auto commandBuffer = queue->BeginRecording();
                auto pass          = m_graph->m_passes[subgraphs[i][j]].get();
                pass->SetActiveFrame(currentFrame);
                pass->withCommandBuffer(commandBuffer, [&pass]() { pass->record(); });

                // TODO: Semaphores & Sync
                if (m_graph->m_subGraphOperatesOnSwapchain[i] && j == subgraphs[i].size() - 1)
                {
                    // Make swapchain image in layout present
                    PipelineBarrier      barrier(m_context, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0);
                    VkImageMemoryBarrier imageBarrier{};
                    imageBarrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imageBarrier.oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
                    imageBarrier.newLayout                   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                    imageBarrier.srcQueueFamilyIndex         = queue->GetQueueFamily();
                    imageBarrier.dstQueueFamilyIndex         = m_swapchain->GetQueueFamily();
                    imageBarrier.image                       = m_swapchain->GetCurrentImage();
                    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    imageBarrier.subresourceRange.levelCount = 1;
                    imageBarrier.subresourceRange.layerCount = 1;
                    imageBarrier.srcAccessMask               = 0;
                    imageBarrier.dstAccessMask               = VK_ACCESS_MEMORY_READ_BIT;
                    barrier.addImageMemoryBarrier(imageBarrier);
                    commandBuffer->AddPipelineBarrier(barrier);

                    TimelineSemaphoreWait wait;
                    wait.m_semaphore = m_swapchain->GetImageAvailableSemaphoreCurrentFrame();
                    wait.m_value     = 0;
                    wait.m_waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

                    auto                  renderFinishSemaphore = m_swapchain->GetRenderingFinishSemaphoreCurrentFrame();

                    auto                  fence = m_swapchain->GetCurrentFrameFence();

                    TimelineSemaphoreWait newWait;
                    if (j == 0)
                    {
                        newWait = queue->SubmitCommand({ wait }, fence, renderFinishSemaphore);
                    }
                    else
                    {
                        newWait = queue->SubmitCommand({ wait, lastWait }, fence, renderFinishSemaphore);
                    }
                    lastWait = newWait;
                }
                else
                {
                    TimelineSemaphoreWait newWait;
                    if (j == 0)
                    {
                        newWait = queue->SubmitCommand({}, nullptr);
                    }
                    else
                    {
                        newWait = queue->SubmitCommand({ lastWait }, nullptr);
                    }
                    lastWait = newWait;
                }
            }
        }
    }

    IFRIT_APIDECL void CommandExecutor::runImmidiateCommand(std::function<void(CommandBuffer*)> func,
        QueueRequirement                                                                        req)
    {
        auto queue              = m_queues;
        auto requiredCapability = getUnderlying(req);
        for (int i = 0; i < queue.size(); i++)
        {
            if ((queue[i]->GetCapability() & requiredCapability) == requiredCapability)
            {
                auto commandBuffer = queue[i]->BeginRecording();
                func(commandBuffer);
                queue[i]->SubmitCommand({}, nullptr);
                queue[i]->WaitIdle();
                return;
            }
        }
        vkrError("No queue found for immediate command");
    }

    IFRIT_APIDECL SwapchainImageResource* CommandExecutor::GetSwapchainImageResource()
    {
        return m_swapchainImageResource.get();
    }

    IFRIT_APIDECL void CommandExecutor::BeginFrame()
    {
        m_swapchain->AcquireNextImage();
    }
    IFRIT_APIDECL void CommandExecutor::EndFrame()
    {
        m_swapchain->Present();
    }

    IFRIT_APIDECL Queue* CommandExecutor::GetQueue(QueueRequirement req)
    {

        // This section is too MESSY
        uint32_t unsel = 0;
        if (req == QueueRequirement::Transfer)
        {
            unsel = getUnderlying(QueueRequirement::Graphics_Compute);
        }
        auto                               requiredCapability = getUnderlying(req);

        std::unordered_map<uint32_t, bool> graphicsQueueList;
        uint32_t                           graphicsQueueCnt = 0;

        for (int i = 0; i < m_queues.size(); i++)
        {
            if ((m_queues[i]->GetCapability()) & getUnderlying(QueueRequirement::Graphics))
            {
                graphicsQueueList[i] = true;
                graphicsQueueCnt++;
                if (graphicsQueueCnt >= m_queuesGraphics.size())
                {
                    break;
                }
            }
        }

        if (req == QueueRequirement::Compute)
        {
            for (int i = 0; i < m_queues.size(); i++)
            {
                if (graphicsQueueList.find(i) != graphicsQueueList.end())
                {
                    continue;
                }
                if ((m_queues[i]->GetCapability() & requiredCapability) == requiredCapability)
                {
                    return m_queues[i];
                }
            }
        }
        else
        {
            for (int i = 0; i < m_queues.size(); i++)
            {
                if ((m_queues[i]->GetCapability() & unsel))
                {
                    continue;
                }
                if ((m_queues[i]->GetCapability() & requiredCapability) == requiredCapability)
                {
                    return m_queues[i];
                }
            }
        }

        std::abort();
        return nullptr;
    }

} // namespace Ifrit::Graphics::VulkanGraphics
