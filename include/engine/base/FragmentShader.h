#pragma once
#include "engine/base/VaryingStore.h"

namespace Ifrit::Engine {
	class FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const VaryingStore* varyings, ifloat4* colorOutput) =0;
		IFRIT_HOST virtual FragmentShader* getCudaClone() { return nullptr; };
	};
}