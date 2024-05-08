#pragma once

namespace Ifrit::Presentation::Window {
	class WindowProvider {
	protected:
		size_t width;
		size_t height;
	public:
		virtual bool setup(size_t width, size_t height) = 0;
		size_t getWidth() const;
		size_t getHeight() const;
	};
}