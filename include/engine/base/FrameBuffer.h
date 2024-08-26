#pragma once
#include "core/data/Image.h"


namespace Ifrit::Engine {
	using Ifrit::Core::Data::ImageU8;
	using Ifrit::Core::Data::ImageF32;


	class IFRIT_APIDECL FrameBuffer {
	private:
		std::vector<std::shared_ptr<ImageF32>> colorAttachment;
		std::shared_ptr<ImageF32> depthAttachment;
		uint32_t width;
		uint32_t height;
	public:
		void setColorAttachments(const std::vector<std::shared_ptr<ImageF32>>& colorAttachment);
		void setDepthAttachment(const std::shared_ptr<ImageF32>& depthAttachment);
		inline ImageF32* getColorAttachment(size_t index){
			return colorAttachment[index].get();
		}
		inline ImageF32* getDepthAttachment() {
			return depthAttachment.get();
		}
		inline uint32_t getWidth() const { return width; }
		inline uint32_t getHeight() const { return height; }
	};
}