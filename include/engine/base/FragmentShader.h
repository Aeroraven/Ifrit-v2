#pragma once
#include "engine/base/FrameBuffer.h"
namespace Ifrit::Engine {
	class FragmentShader {
	public:
		virtual void execute(const std::vector<std::any>& varyings, std::vector<float4>& colorOutput) =0;

	};
}