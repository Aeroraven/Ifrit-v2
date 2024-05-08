#pragma once
#include "core/data/Image.h"

namespace Ifrit::Engine {
	using Ifrit::Core::Data::ImageU8;
	using Ifrit::Core::Data::ImageF32;

	class FrameBuffer {
	private:
		std::vector<std::shared_ptr<ImageU8>> colorAttachment;
		std::shared_ptr<ImageF32> depthAttachment;
	public:
		void setColorAttachments(const std::vector<std::shared_ptr<ImageU8>>& colorAttachment);
		void setDepthAttachment(const std::shared_ptr<ImageF32>& depthAttachment);
		ImageU8* getColorAttachment(size_t index);
		ImageF32* getDepthAttachment();
	};
}