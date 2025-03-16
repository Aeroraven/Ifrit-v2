
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
#include "ifrit/softgraphics/core/data/Image.h"

namespace Ifrit::GraphicsBackend::SoftGraphics {
using Ifrit::GraphicsBackend::SoftGraphics::Core::Data::ImageF32;
using Ifrit::GraphicsBackend::SoftGraphics::Core::Data::ImageU8;

struct FrameBufferContext {
  std::vector<ImageF32 *> colorAttachment;
  ImageF32 *depthAttachment;
};

class IFRIT_APIDECL FrameBuffer {
private:
  FrameBufferContext *context;
  u32 width;
  u32 height;

public:
  FrameBuffer();
  ~FrameBuffer();
  void setColorAttachments(const std::vector<ImageF32 *> &colorAttachment);
  void setDepthAttachment(ImageF32 &depthAttachment);

  /* Inline */
  inline ImageF32 *getColorAttachment(size_t index) { return context->colorAttachment[index]; }
  inline ImageF32 *getDepthAttachment() { return context->depthAttachment; }
  inline u32 getWidth() const { return width; }
  inline u32 getHeight() const { return height; }

  /* DLL Compat*/
  void setColorAttachmentsCompatible(ImageF32 *const *colorAttachments, int nums);
  void setDepthAttachmentCompatible(ImageF32 *depthAttachment);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics