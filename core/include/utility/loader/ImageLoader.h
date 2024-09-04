#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Utility::Loader {
	class ImageLoader {
	public:
		void loadRGBA(const char* fileName, std::vector<float>* bufferOut, int* height, int* width);
	};
}