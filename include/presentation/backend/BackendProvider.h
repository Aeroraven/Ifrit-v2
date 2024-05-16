#pragma once
#include "core/definition/CoreExports.h"
#include "core/data/Image.h"
namespace Ifrit::Presentation::Backend {
	class BackendProvider {
	public:
		virtual void draw() = 0;
		virtual void updateTexture(const Ifrit::Core::Data::ImageF32& image) = 0;
	};
}