
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

#include "RhiBaseTypes.h"

namespace Ifrit::GraphicsBackend::Rhi {
struct IFRIT_APIDECL RhiRenderTargetsFormat {
  RhiImageFormat m_depthFormat;
  Vec<RhiImageFormat> m_colorFormats;
};

class IFRIT_APIDECL RhiRenderTargets {
public:
  virtual void setColorAttachments(const Vec<RhiColorAttachment *> &attachments) = 0;
  virtual void setDepthStencilAttachment(RhiDepthStencilAttachment *attachment) = 0;
  virtual void beginRendering(const RhiCommandList *commandBuffer) const = 0;
  virtual void endRendering(const RhiCommandList *commandBuffer) const = 0;
  virtual void setRenderArea(RhiScissor area) = 0;
  virtual RhiRenderTargetsFormat getFormat() const = 0;
  virtual RhiScissor getRenderArea() const = 0;
  virtual RhiDepthStencilAttachment *getDepthStencilAttachment() const = 0;
  virtual RhiColorAttachment *getColorAttachment(u32 index) const = 0;
};

class IFRIT_APIDECL RhiColorAttachment {
public:
  virtual RhiTexture *getRenderTarget() const = 0;
  virtual void setBlendInfo(const RhiAttachmentBlendInfo &info) = 0;
};

class IFRIT_APIDECL RhiDepthStencilAttachment {
public:
  virtual RhiTexture *getTexture() const = 0;
};

class IFRIT_APIDECL RhiVertexBufferView {
protected:
  virtual void addBinding(Vec<u32> location, Vec<RhiImageFormat> format, Vec<u32> offset, u32 stride,
                          RhiVertexInputRate inputRate = RhiVertexInputRate::Vertex) = 0;
};
} // namespace Ifrit::GraphicsBackend::Rhi