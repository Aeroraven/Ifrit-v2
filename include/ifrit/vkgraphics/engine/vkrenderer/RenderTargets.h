
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

#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"

namespace Ifrit::GraphicsBackend::VulkanGraphics {
class IFRIT_APIDECL ColorAttachment : public Rhi::RhiColorAttachment {
private:
  SingleDeviceImage *m_renderTarget;
  Rhi::RhiClearValue m_clearValue;
  Rhi::RhiRenderTargetLoadOp m_loadOp;
  u32 m_targetMip = ~0u;
  u32 m_targetArrLayer = ~0u;
  Rhi::RhiAttachmentBlendInfo m_blendInfo;

public:
  ColorAttachment(Rhi::RhiTexture *renderTarget, Rhi::RhiClearValue clearValue, Rhi::RhiRenderTargetLoadOp loadOp,
                  u32 mip, u32 arrLayer)
      : m_renderTarget(Ifrit::Common::Utility::checked_cast<SingleDeviceImage>(renderTarget)), m_clearValue(clearValue),
        m_loadOp(loadOp), m_targetMip(mip), m_targetArrLayer(arrLayer) {}

  inline SingleDeviceImage *getRenderTargetInternal() const { return m_renderTarget; }
  inline Rhi::RhiTexture *getRenderTarget() const override { return m_renderTarget; }
  inline Rhi::RhiClearValue getClearValue() const { return m_clearValue; }
  inline Rhi::RhiRenderTargetLoadOp getLoadOp() const { return m_loadOp; }
  inline u32 getTargetMip() const { return m_targetMip; }
  inline u32 getTargetArrLayer() const { return m_targetArrLayer; }

  inline void setBlendInfo(const Rhi::RhiAttachmentBlendInfo &info) override { m_blendInfo = info; }
  inline Rhi::RhiAttachmentBlendInfo getBlendInfo() const { return m_blendInfo; }
};

class IFRIT_APIDECL DepthStencilAttachment : public Rhi::RhiDepthStencilAttachment {
private:
  SingleDeviceImage *m_renderTarget;
  Rhi::RhiClearValue m_clearValue;
  Rhi::RhiRenderTargetLoadOp m_loadOp;

public:
  DepthStencilAttachment(Rhi::RhiTexture *renderTarget, Rhi::RhiClearValue clearValue,
                         Rhi::RhiRenderTargetLoadOp loadOp)
      : m_renderTarget(Ifrit::Common::Utility::checked_cast<SingleDeviceImage>(renderTarget)), m_clearValue(clearValue),
        m_loadOp(loadOp) {}

  inline SingleDeviceImage *getRenderTargetInternal() const { return m_renderTarget; }
  inline Rhi::RhiTexture *getRenderTarget() const { return m_renderTarget; }
  inline Rhi::RhiTexture *getTexture() const override { return m_renderTarget; }
  inline Rhi::RhiClearValue getClearValue() const { return m_clearValue; }
  inline Rhi::RhiRenderTargetLoadOp getLoadOp() const { return m_loadOp; }
};

class IFRIT_APIDECL RenderTargets : public Rhi::RhiRenderTargets {
private:
  Vec<ColorAttachment *> m_colorAttachments;
  DepthStencilAttachment *m_depthStencilAttachment = nullptr;
  EngineContext *m_context;
  Rhi::RhiScissor m_renderArea;

public:
  RenderTargets(EngineContext *context) : m_context(context) {}
  ~RenderTargets() = default;

  inline void setRenderArea(Rhi::RhiScissor area) override { m_renderArea = area; }
  void setColorAttachments(const Vec<Rhi::RhiColorAttachment *> &attachments) override;
  void setDepthStencilAttachment(Rhi::RhiDepthStencilAttachment *attachment) override;
  void beginRendering(const Rhi::RhiCommandList *commandBuffer) const override;
  void endRendering(const Rhi::RhiCommandList *commandBuffer) const override;
  Rhi::RhiRenderTargetsFormat getFormat() const override;
  virtual Rhi::RhiScissor getRenderArea() const override;
  inline Rhi::RhiDepthStencilAttachment *getDepthStencilAttachment() const override { return m_depthStencilAttachment; }
  inline Rhi::RhiColorAttachment *getColorAttachment(u32 index) const { return m_colorAttachments[index]; }
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics