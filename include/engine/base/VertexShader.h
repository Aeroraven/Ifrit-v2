#pragma once
#include "engine/base/VaryingStore.h"

namespace Ifrit::Engine {
	class VertexShader{
	public:
		IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, VaryingStore** outVaryings) {
			printf("Base\n");
		};
	};
}