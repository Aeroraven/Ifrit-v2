
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


#include "ifrit/softgraphics/engine/base/FrameBuffer.h"
#include "ifrit/common/util/TypingUtil.h"
using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::SoftGraphics {
IFRIT_APIDECL void FrameBuffer::setColorAttachments(
    const std::vector<ImageF32 *> &colorAttachment) {
  this->context->colorAttachment = colorAttachment;
  this->width = size_cast<uint32_t>(colorAttachment[0]->getWidth());
  this->height = size_cast<uint32_t>(colorAttachment[0]->getHeight());
}
IFRIT_APIDECL void FrameBuffer::setDepthAttachment(ImageF32 &depthAttachment) {
  this->context->depthAttachment = &depthAttachment;
}

IFRIT_APIDECL FrameBuffer::FrameBuffer() {
  this->context = new std::remove_pointer_t<decltype(this->context)>();
}
IFRIT_APIDECL FrameBuffer::~FrameBuffer() { delete this->context; }

/* DLL Compat */

IFRIT_APIDECL void
FrameBuffer::setColorAttachmentsCompatible(ImageF32 *const *colorAttachment,
                                           int nums) {
  this->context->colorAttachment = std::vector<ImageF32 *>(nums);
  for (int i = 0; i < nums; i++) {
    this->context->colorAttachment[i] = colorAttachment[i];
  }
  this->width = size_cast<uint32_t>(colorAttachment[0]->getWidth());
  this->height = size_cast<uint32_t>(colorAttachment[0]->getHeight());
}
IFRIT_APIDECL void
FrameBuffer::setDepthAttachmentCompatible(ImageF32 *depthAttachment) {
  this->context->depthAttachment = depthAttachment;
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics