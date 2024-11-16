#include "ifrit/vkgraphics/engine/vkrenderer/RenderTargets.h"
#include "ifrit/common/util/TypingUtil.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::VulkanGraphics {

Rhi::RhiImageFormat toRhiFormat(VkFormat rawFormat) {
  return static_cast<Rhi::RhiImageFormat>(rawFormat);
}

IFRIT_APIDECL void RenderTargets::setColorAttachments(
    const std::vector<Rhi::RhiColorAttachment *> &attachments) {
  m_colorAttachments.clear();
  for (auto attachment : attachments) {
    m_colorAttachments.push_back(static_cast<ColorAttachment *>(attachment));
  }
}

IFRIT_APIDECL void RenderTargets::setDepthStencilAttachment(
    Rhi::RhiDepthStencilAttachment *attachment) {
  m_depthStencilAttachment = static_cast<DepthStencilAttachment *>(attachment);
}

IFRIT_APIDECL void RenderTargets::beginRendering(
    const Rhi::RhiCommandBuffer *commandBuffer) const {
  auto cmd = checked_cast<CommandBuffer>(commandBuffer);
  auto cmdraw = cmd->getCommandBuffer();

  // other kind of layout transition should be handled by the
  // render graph or explicitly by the user.
  auto depthSrcLayout = (m_depthStencilAttachment->getLoadOp() ==
                         Rhi::RhiRenderTargetLoadOp::Clear)
                            ? Rhi::RhiResourceState::Undefined
                            : Rhi::RhiResourceState::DepthStencilRenderTarget;
  cmd->imageBarrier(m_depthStencilAttachment->getRenderTarget(), depthSrcLayout,
                    Rhi::RhiResourceState::DepthStencilRenderTarget,
                    {0, 0, 1, 1});

  for (auto attachment : m_colorAttachments) {
    auto srcLayout =
        (attachment->getLoadOp() == Rhi::RhiRenderTargetLoadOp::Clear)
            ? Rhi::RhiResourceState::Undefined
            : Rhi::RhiResourceState::RenderTarget;
    cmd->imageBarrier(attachment->getRenderTarget(), srcLayout,
                      Rhi::RhiResourceState::RenderTarget, {0, 0, 1, 1});
  }
  auto exfunc = m_context->getExtensionFunction();

  // Specify rendering info
  std::vector<VkRenderingAttachmentInfoKHR> colorAttachmentInfos;
  VkRenderingAttachmentInfoKHR depthAttachmentInfo{};
  if (m_depthStencilAttachment) {
    VkClearValue clearValue;
    clearValue.depthStencil.depth =
        m_depthStencilAttachment->getClearValue().m_depth;
    clearValue.depthStencil.stencil =
        m_depthStencilAttachment->getClearValue().m_stencil;

    VkAttachmentLoadOp loadOp;
    if (m_depthStencilAttachment->getLoadOp() ==
        Rhi::RhiRenderTargetLoadOp::Clear) {
      loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    } else if (m_depthStencilAttachment->getLoadOp() ==
               Rhi::RhiRenderTargetLoadOp::DontCare) {
      loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    } else {
      loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    }

    depthAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    depthAttachmentInfo.clearValue = clearValue;
    depthAttachmentInfo.loadOp = loadOp;
    depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    depthAttachmentInfo.imageView =
        m_depthStencilAttachment->getRenderTargetInternal()->getImageView();
  }
  for (auto attachment : m_colorAttachments) {
    VkClearValue clearValue;
    clearValue.color.float32[0] = attachment->getClearValue().m_color[0];
    clearValue.color.float32[1] = attachment->getClearValue().m_color[1];
    clearValue.color.float32[2] = attachment->getClearValue().m_color[2];
    clearValue.color.float32[3] = attachment->getClearValue().m_color[3];

    VkAttachmentLoadOp loadOp;
    if (attachment->getLoadOp() == Rhi::RhiRenderTargetLoadOp::Clear) {
      loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    } else if (attachment->getLoadOp() ==
               Rhi::RhiRenderTargetLoadOp::DontCare) {
      loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    } else {
      loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    }
    auto tgtMip = attachment->getTargetMip();
    auto tgtArrLayer = attachment->getTargetArrLayer();

    VkRenderingAttachmentInfoKHR attachmentInfo{};
    attachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    attachmentInfo.clearValue = clearValue;
    attachmentInfo.loadOp = loadOp;
    attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    attachmentInfo.imageView =
        attachment->getRenderTargetInternal()->getImageViewMipLayer(
            tgtMip, tgtArrLayer, 1, 1);
    colorAttachmentInfos.push_back(attachmentInfo);
  }
  VkRect2D renderArea;
  renderArea.extent = {m_renderArea.width, m_renderArea.height};
  renderArea.offset = {m_renderArea.x, m_renderArea.y};

  VkRenderingInfo renderingInfo{};
  renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  renderingInfo.renderArea = renderArea;
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount =
      size_cast<uint32_t>(m_colorAttachments.size());
  renderingInfo.pColorAttachments = colorAttachmentInfos.data();
  renderingInfo.pDepthAttachment =
      m_depthStencilAttachment ? &depthAttachmentInfo : nullptr;

  vkCmdBeginRendering(cmdraw, &renderingInfo);

  // setup dynamic rendering info
  std::vector<VkBool32> colorWrite;
  std::vector<VkBool32> blendEnable;
  std::vector<VkColorBlendEquationEXT> blendEquations;
  std::vector<VkColorComponentFlags> colorWriteMask;
  for (auto attachment : m_colorAttachments) {
    colorWrite.push_back(VK_TRUE);
    blendEnable.push_back(VK_FALSE);
    VkColorBlendEquationEXT colorBlendEquation{};
    colorBlendEquation.alphaBlendOp = VK_BLEND_OP_ADD;
    colorBlendEquation.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendEquation.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendEquation.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendEquation.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendEquation.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendEquations.push_back(colorBlendEquation);
    colorWriteMask.push_back(
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
  }
  exfunc.p_vkCmdSetColorWriteEnableEXT(
      cmdraw, size_cast<uint32_t>(m_colorAttachments.size()),
      colorWrite.data());
  exfunc.p_vkCmdSetColorBlendEnableEXT(
      cmdraw, 0, size_cast<uint32_t>(m_colorAttachments.size()),
      blendEnable.data());
  exfunc.p_vkCmdSetColorBlendEquationEXT(
      cmdraw, 0, size_cast<uint32_t>(m_colorAttachments.size()),
      blendEquations.data());
  exfunc.p_vkCmdSetColorWriteMaskEXT(
      cmdraw, 0, size_cast<uint32_t>(m_colorAttachments.size()),
      colorWriteMask.data());

  // Set default viewport & scissor
  VkViewport viewport{};
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = m_renderArea.width;
  viewport.height = m_renderArea.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {m_renderArea.width, m_renderArea.height};

  vkCmdSetViewport(cmdraw, 0, 1, &viewport);
  vkCmdSetScissor(cmdraw, 0, 1, &scissor);

  // If depth attachment presents, we assume that depth test is required
  if (m_depthStencilAttachment) {
    auto extf = m_context->getExtensionFunction();

    extf.p_vkCmdSetDepthTestEnable(cmdraw, VK_TRUE);
    extf.p_vkCmdSetDepthWriteEnable(cmdraw, VK_TRUE);
    extf.p_vkCmdSetDepthCompareOp(cmdraw, VK_COMPARE_OP_LESS);
  }
}

IFRIT_APIDECL void
RenderTargets::endRendering(const Rhi::RhiCommandBuffer *commandBuffer) const {
  auto cmd = checked_cast<CommandBuffer>(commandBuffer);
  auto cmdraw = cmd->getCommandBuffer();
  vkCmdEndRendering(cmdraw);
}

IFRIT_APIDECL Rhi::RhiRenderTargetsFormat RenderTargets::getFormat() const {
  Rhi::RhiRenderTargetsFormat format;
  if (m_depthStencilAttachment) {
    format.m_depthFormat = toRhiFormat(
        m_depthStencilAttachment->getRenderTargetInternal()->getFormat());
  } else {
    format.m_depthFormat = Rhi::RhiImageFormat::RHI_FORMAT_UNDEFINED;
  }
  for (auto attachment : m_colorAttachments) {
    format.m_colorFormats.push_back(
        toRhiFormat(attachment->getRenderTargetInternal()->getFormat()));
  }
  return format;
}

IFRIT_APIDECL Rhi::RhiScissor RenderTargets::getRenderArea() const {
  return m_renderArea;
}

} // namespace Ifrit::GraphicsBackend::VulkanGraphics