#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"

namespace Ifrit::GraphicsBackend::VulkanGraphics {
class IFRIT_APIDECL ColorAttachment : public Rhi::RhiColorAttachment {
private:
  SingleDeviceImage *m_renderTarget;
  Rhi::RhiClearValue m_clearValue;
  Rhi::RhiRenderTargetLoadOp m_loadOp;
  uint32_t m_targetMip = ~0u;
  uint32_t m_targetArrLayer = ~0u;

public:
  ColorAttachment(Rhi::RhiTexture *renderTarget, Rhi::RhiClearValue clearValue,
                  Rhi::RhiRenderTargetLoadOp loadOp, uint32_t mip,
                  uint32_t arrLayer)
      : m_renderTarget(Ifrit::Common::Utility::checked_cast<SingleDeviceImage>(
            renderTarget)),
        m_clearValue(clearValue), m_loadOp(loadOp), m_targetMip(mip),
        m_targetArrLayer(arrLayer) {}

  inline SingleDeviceImage *getRenderTargetInternal() const {
    return m_renderTarget;
  }
  inline Rhi::RhiTexture *getRenderTarget() const override {
    return m_renderTarget;
  }
  inline Rhi::RhiClearValue getClearValue() const { return m_clearValue; }
  inline Rhi::RhiRenderTargetLoadOp getLoadOp() const { return m_loadOp; }
  inline uint32_t getTargetMip() const { return m_targetMip; }
  inline uint32_t getTargetArrLayer() const { return m_targetArrLayer; }
};

class IFRIT_APIDECL DepthStencilAttachment
    : public Rhi::RhiDepthStencilAttachment {
private:
  SingleDeviceImage *m_renderTarget;
  Rhi::RhiClearValue m_clearValue;
  Rhi::RhiRenderTargetLoadOp m_loadOp;

public:
  DepthStencilAttachment(Rhi::RhiTexture *renderTarget,
                         Rhi::RhiClearValue clearValue,
                         Rhi::RhiRenderTargetLoadOp loadOp)
      : m_renderTarget(Ifrit::Common::Utility::checked_cast<SingleDeviceImage>(
            renderTarget)),
        m_clearValue(clearValue), m_loadOp(loadOp) {}

  inline SingleDeviceImage *getRenderTargetInternal() const {
    return m_renderTarget;
  }
  inline Rhi::RhiTexture *getRenderTarget() const { return m_renderTarget; }
  inline Rhi::RhiTexture *getTexture() const override { return m_renderTarget; }
  inline Rhi::RhiClearValue getClearValue() const { return m_clearValue; }
  inline Rhi::RhiRenderTargetLoadOp getLoadOp() const { return m_loadOp; }
};

class IFRIT_APIDECL RenderTargets : public Rhi::RhiRenderTargets {
private:
  std::vector<ColorAttachment *> m_colorAttachments;
  DepthStencilAttachment *m_depthStencilAttachment = nullptr;
  EngineContext *m_context;
  Rhi::RhiScissor m_renderArea;

public:
  RenderTargets(EngineContext *context) : m_context(context) {}
  ~RenderTargets() = default;

  inline void setRenderArea(Rhi::RhiScissor area) override {
    m_renderArea = area;
  }
  void setColorAttachments(
      const std::vector<Rhi::RhiColorAttachment *> &attachments) override;
  void setDepthStencilAttachment(
      Rhi::RhiDepthStencilAttachment *attachment) override;
  void
  beginRendering(const Rhi::RhiCommandBuffer *commandBuffer) const override;
  void endRendering(const Rhi::RhiCommandBuffer *commandBuffer) const override;
  Rhi::RhiRenderTargetsFormat getFormat() const override;
  virtual Rhi::RhiScissor getRenderArea() const override;
  inline Rhi::RhiDepthStencilAttachment *
  getDepthStencilAttachment() const override {
    return m_depthStencilAttachment;
  }
  inline Rhi::RhiColorAttachment *getColorAttachment(uint32_t index) const {
    return m_colorAttachments[index];
  }
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics