#pragma once
#include "core/data/Image.h"

namespace Ifrit::Engine {
using Ifrit::Core::Data::ImageF32;
using Ifrit::Core::Data::ImageU8;

struct FrameBufferContext {
  std::vector<ImageF32 *> colorAttachment;
  ImageF32 *depthAttachment;
};

class IFRIT_APIDECL FrameBuffer {
private:
  FrameBufferContext *context;
  uint32_t width;
  uint32_t height;

public:
  FrameBuffer();
  ~FrameBuffer();
  void setColorAttachments(const std::vector<ImageF32 *> &colorAttachment);
  void setDepthAttachment(ImageF32 &depthAttachment);

  /* Inline */
  inline ImageF32 *getColorAttachment(size_t index) {
    return context->colorAttachment[index];
  }
  inline ImageF32 *getDepthAttachment() { return context->depthAttachment; }
  inline uint32_t getWidth() const { return width; }
  inline uint32_t getHeight() const { return height; }

  /* DLL Compat*/
  void setColorAttachmentsCompatible(ImageF32 *const *colorAttachments,
                                     int nums);
  void setDepthAttachmentCompatible(ImageF32 *depthAttachment);
};
} // namespace Ifrit::Engine