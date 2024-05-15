#pragma once
#include "engine/base/FrameBuffer.h"
#include "engine/base/VaryingStore.h"

namespace Ifrit::Engine {
	class FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const VaryingStore* varyings, float4* colorOutput) =0;

	};
}