#include "./engine/base/FrameBuffer.h"

namespace Ifrit::Engine {
	void FrameBuffer::setColorAttachments(const std::vector<std::shared_ptr<ImageU8>>& colorAttachment) {
		this->colorAttachment = colorAttachment;
	}
	void FrameBuffer::setDepthAttachment(const std::shared_ptr<ImageF32>& depthAttachment) {
		this->depthAttachment = depthAttachment;
	}
	ImageU8* FrameBuffer::getColorAttachment(size_t index) {
		ifritAssert(index < colorAttachment.size(), "Index out of range");
		return colorAttachment[index].get();
	}
	ImageF32* FrameBuffer::getDepthAttachment() {
		return depthAttachment.get();
	}
}