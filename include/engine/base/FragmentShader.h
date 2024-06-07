#pragma once
#include "engine/base/VaryingStore.h"

#define ifritGetOutputColorPtr(x) ((stride)?(reinterpret_cast<ifloat4s256*>(x)):(reinterpret_cast<ifloat4*>(x)))

namespace Ifrit::Engine {
	class FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const void* varyings, void* colorOutput, int stride) =0;
		IFRIT_HOST virtual FragmentShader* getCudaClone() { return nullptr; };
	};
}