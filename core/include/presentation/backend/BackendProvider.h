#pragma once
#include "core/definition/CoreExports.h"
#include "core/data/Image.h"
namespace Ifrit::Presentation::Backend {
	class BackendProvider {
	public:
		virtual ~BackendProvider() = default;
		virtual void draw() = 0;
		virtual void updateTexture(const Ifrit::Core::Data::ImageF32& image) = 0;
		virtual void setViewport(int32_t x, int32_t y, int32_t width, int32_t height) = 0;
	};
}