#pragma once
#include "engine/base/FrameBuffer.h"
#include "engine/base/VaryingStore.h"

namespace Ifrit::Engine {
	class FragmentShader {
	public:
		virtual void execute(const std::vector<VaryingStore>& varyings, std::vector<float4>& colorOutput) =0;

	};
}