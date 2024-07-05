#pragma once
#include "engine/base/Constants.h"
namespace Ifrit::Engine {
	struct IfritSamplerT {
		IfritFilter filterMode = IF_FILTER_NEAREST;
		IfritSamplerAddressMode addressModeU = IF_SAMPLER_ADDRESS_MODE_REPEAT;
		IfritSamplerAddressMode addressModeV = IF_SAMPLER_ADDRESS_MODE_REPEAT;
		IfritSamplerAddressMode addressModeW = IF_SAMPLER_ADDRESS_MODE_REPEAT;
		IfritBorderColor borderColor = IF_BORDER_COLOR_BLACK;
	};
}