#include "engine/base/FrameBuffer.h"

namespace Ifrit::Engine {
	void FrameBuffer::setColorAttachments(const std::vector<std::shared_ptr<ImageF32>>& colorAttachment) {
		this->colorAttachment = colorAttachment;
		this->width = colorAttachment[0]->getWidth();
		this->height = colorAttachment[0]->getHeight();
	}
	void FrameBuffer::setDepthAttachment(const std::shared_ptr<ImageF32>& depthAttachment) {
		this->depthAttachment = depthAttachment;
	}
	
}