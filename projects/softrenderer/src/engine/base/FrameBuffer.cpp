#include "engine/base/FrameBuffer.h"

namespace Ifrit::Engine {
IFRIT_APIDECL void FrameBuffer::setColorAttachments(
    const std::vector<ImageF32 *> &colorAttachment) {
  this->context->colorAttachment = colorAttachment;
  this->width = colorAttachment[0]->getWidth();
  this->height = colorAttachment[0]->getHeight();
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
  this->width = colorAttachment[0]->getWidth();
  this->height = colorAttachment[0]->getHeight();
}
IFRIT_APIDECL void
FrameBuffer::setDepthAttachmentCompatible(ImageF32 *depthAttachment) {
  this->context->depthAttachment = depthAttachment;
}
} // namespace Ifrit::Engine