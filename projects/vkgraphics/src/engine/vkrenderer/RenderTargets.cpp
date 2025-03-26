
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

#include "ifrit/vkgraphics/engine/vkrenderer/RenderTargets.h"
#include "ifrit/common/util/TypingUtil.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::Graphics::VulkanGraphics
{

    Rhi::RhiImageFormat toRhiFormat(VkFormat rawFormat)
    {
        return static_cast<Rhi::RhiImageFormat>(rawFormat);
    }

    IFRIT_APIDECL void RenderTargets::SetColorAttachments(const Vec<Rhi::RhiColorAttachment*>& attachments)
    {
        m_colorAttachments.clear();
        for (auto attachment : attachments)
        {
            m_colorAttachments.push_back(static_cast<ColorAttachment*>(attachment));
        }
    }

    IFRIT_APIDECL void RenderTargets::SetDepthStencilAttachment(Rhi::RhiDepthStencilAttachment* attachment)
    {
        m_depthStencilAttachment = static_cast<DepthStencilAttachment*>(attachment);
    }

    IFRIT_APIDECL void RenderTargets::BeginRendering(const Rhi::RhiCommandList* commandBuffer) const
    {
        auto cmd    = CheckedCast<CommandBuffer>(commandBuffer);
        auto cmdraw = cmd->GetCommandBuffer();

        // other kind of layout transition should be handled by the
        // render graph or explicitly by the user.
        if (m_depthStencilAttachment != nullptr)
        {
            auto depthSrcLayout = (m_depthStencilAttachment->GetLoadOp() == Rhi::RhiRenderTargetLoadOp::Clear)
                ? Rhi::RhiResourceState::Undefined
                : Rhi::RhiResourceState::DepthStencilRT;
            cmd->AddImageBarrier(m_depthStencilAttachment->GetRenderTarget(), depthSrcLayout,
                Rhi::RhiResourceState::DepthStencilRT, { 0, 0, 1, 1 });
        }

        for (auto attachment : m_colorAttachments)
        {
            auto srcLayout = (attachment->GetLoadOp() == Rhi::RhiRenderTargetLoadOp::Clear) ? Rhi::RhiResourceState::Undefined
                                                                                            : Rhi::RhiResourceState::ColorRT;
            cmd->AddImageBarrier(attachment->GetRenderTarget(), srcLayout, Rhi::RhiResourceState::ColorRT, { 0, 0, 1, 1 });
        }
        auto                              exfunc = m_context->GetExtensionFunction();

        // Specify rendering info
        Vec<VkRenderingAttachmentInfoKHR> colorAttachmentInfos;
        VkRenderingAttachmentInfoKHR      depthAttachmentInfo{};
        if (m_depthStencilAttachment)
        {
            VkClearValue clearValue;
            clearValue.depthStencil.depth   = m_depthStencilAttachment->GetClearValue().m_depth;
            clearValue.depthStencil.stencil = m_depthStencilAttachment->GetClearValue().m_stencil;

            VkAttachmentLoadOp loadOp;
            if (m_depthStencilAttachment->GetLoadOp() == Rhi::RhiRenderTargetLoadOp::Clear)
            {
                loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            }
            else if (m_depthStencilAttachment->GetLoadOp() == Rhi::RhiRenderTargetLoadOp::DontCare)
            {
                loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            }
            else
            {
                loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            }

            depthAttachmentInfo.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
            depthAttachmentInfo.clearValue  = clearValue;
            depthAttachmentInfo.loadOp      = loadOp;
            depthAttachmentInfo.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            depthAttachmentInfo.imageView   = m_depthStencilAttachment->GetRenderTargetInternal()->GetImageView();
        }
        for (auto attachment : m_colorAttachments)
        {
            VkClearValue clearValue;
            clearValue.color.float32[0] = attachment->GetClearValue().m_color[0];
            clearValue.color.float32[1] = attachment->GetClearValue().m_color[1];
            clearValue.color.float32[2] = attachment->GetClearValue().m_color[2];
            clearValue.color.float32[3] = attachment->GetClearValue().m_color[3];

            VkAttachmentLoadOp loadOp;
            if (attachment->GetLoadOp() == Rhi::RhiRenderTargetLoadOp::Clear)
            {
                loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            }
            else if (attachment->GetLoadOp() == Rhi::RhiRenderTargetLoadOp::DontCare)
            {
                loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            }
            else
            {
                loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            }
            auto                         tgtMip      = attachment->GetTargetMip();
            auto                         tgtArrLayer = attachment->GetTargetArrLayer();

            VkRenderingAttachmentInfoKHR attachmentInfo{};
            attachmentInfo.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
            attachmentInfo.clearValue  = clearValue;
            attachmentInfo.loadOp      = loadOp;
            attachmentInfo.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            attachmentInfo.imageView   = attachment->GetRenderTargetInternal()->GetImageViewMipLayer(tgtMip, tgtArrLayer, 1, 1);
            colorAttachmentInfos.push_back(attachmentInfo);
        }
        VkRect2D renderArea;
        renderArea.extent = { m_renderArea.width, m_renderArea.height };
        renderArea.offset = { m_renderArea.x, m_renderArea.y };

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea           = renderArea;
        renderingInfo.layerCount           = 1;
        renderingInfo.colorAttachmentCount = SizeCast<uint32_t>(m_colorAttachments.size());
        renderingInfo.pColorAttachments    = colorAttachmentInfos.data();
        renderingInfo.pDepthAttachment     = m_depthStencilAttachment ? &depthAttachmentInfo : nullptr;

        vkCmdBeginRendering(cmdraw, &renderingInfo);

        // setup dynamic rendering info
        Vec<VkBool32>                colorWrite;
        Vec<VkBool32>                blendEnable;
        Vec<VkColorBlendEquationEXT> blendEquations;
        Vec<VkColorComponentFlags>   colorWriteMask;

#define TOVKBLENDOP(op) static_cast<VkBlendOp>(op)
#define TOVKBLENDFACTOR(factor) static_cast<VkBlendFactor>(factor)

        for (auto attachment : m_colorAttachments)
        {
            auto blendInfo = attachment->GetBlendInfo();
            colorWrite.push_back(VK_TRUE);
            blendEnable.push_back(blendInfo.m_blendEnable ? VK_TRUE : VK_FALSE);
            VkColorBlendEquationEXT colorBlendEquation{};
            colorBlendEquation.alphaBlendOp        = TOVKBLENDOP(blendInfo.m_alphaBlendOp);
            colorBlendEquation.colorBlendOp        = TOVKBLENDOP(blendInfo.m_colorBlendOp);
            colorBlendEquation.dstAlphaBlendFactor = TOVKBLENDFACTOR(blendInfo.m_dstAlphaBlendFactor);
            colorBlendEquation.dstColorBlendFactor = TOVKBLENDFACTOR(blendInfo.m_dstColorBlendFactor);
            colorBlendEquation.srcAlphaBlendFactor = TOVKBLENDFACTOR(blendInfo.m_srcAlphaBlendFactor);
            colorBlendEquation.srcColorBlendFactor = TOVKBLENDFACTOR(blendInfo.m_srcColorBlendFactor);
            blendEquations.push_back(colorBlendEquation);
            colorWriteMask.push_back(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
        }
#undef TOVKBLENDOP
#undef TOVKBLENDFACTOR
        if (m_colorAttachments.size() > 0)
        {
            exfunc.p_vkCmdSetColorWriteEnableEXT(cmdraw, SizeCast<uint32_t>(m_colorAttachments.size()), colorWrite.data());
            exfunc.p_vkCmdSetColorBlendEnableEXT(cmdraw, 0, SizeCast<uint32_t>(m_colorAttachments.size()), blendEnable.data());
            exfunc.p_vkCmdSetColorBlendEquationEXT(cmdraw, 0, SizeCast<uint32_t>(m_colorAttachments.size()),
                blendEquations.data());
            exfunc.p_vkCmdSetColorWriteMaskEXT(cmdraw, 0, SizeCast<uint32_t>(m_colorAttachments.size()), colorWriteMask.data());
        }

        // Set default viewport & scissor
        VkViewport viewport{};
        viewport.x        = 0;
        viewport.y        = 0;
        viewport.width    = static_cast<float>(m_renderArea.width);
        viewport.height   = static_cast<float>(m_renderArea.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = { m_renderArea.width, m_renderArea.height };

        vkCmdSetViewport(cmdraw, 0, 1, &viewport);
        vkCmdSetScissor(cmdraw, 0, 1, &scissor);

        // If depth attachment presents, we assume that depth test is required
        if (m_depthStencilAttachment)
        {
            auto extf = m_context->GetExtensionFunction();

            extf.p_vkCmdSetDepthTestEnable(cmdraw, VK_TRUE);
            extf.p_vkCmdSetDepthWriteEnable(cmdraw, VK_TRUE);
            extf.p_vkCmdSetDepthCompareOp(cmdraw, VK_COMPARE_OP_LESS);
        }
        else
        {
            auto extf = m_context->GetExtensionFunction();
            extf.p_vkCmdSetDepthTestEnable(cmdraw, VK_FALSE);
            extf.p_vkCmdSetDepthWriteEnable(cmdraw, VK_FALSE);
            extf.p_vkCmdSetDepthCompareOp(cmdraw, VK_COMPARE_OP_LESS);
        }
    }

    IFRIT_APIDECL void RenderTargets::EndRendering(const Rhi::RhiCommandList* commandBuffer) const
    {
        auto cmd    = CheckedCast<CommandBuffer>(commandBuffer);
        auto cmdraw = cmd->GetCommandBuffer();
        vkCmdEndRendering(cmdraw);
    }

    IFRIT_APIDECL Rhi::RhiRenderTargetsFormat RenderTargets::GetFormat() const
    {
        Rhi::RhiRenderTargetsFormat format;
        if (m_depthStencilAttachment)
        {
            format.m_depthFormat = toRhiFormat(m_depthStencilAttachment->GetRenderTargetInternal()->GetFormat());
        }
        else
        {
            format.m_depthFormat = Rhi::RhiImageFormat::RhiImgFmt_UNDEFINED;
        }
        for (auto attachment : m_colorAttachments)
        {
            format.m_colorFormats.push_back(toRhiFormat(attachment->GetRenderTargetInternal()->GetFormat()));
        }
        return format;
    }

    IFRIT_APIDECL Rhi::RhiScissor RenderTargets::GetRenderArea() const
    {
        return m_renderArea;
    }

} // namespace Ifrit::Graphics::VulkanGraphics