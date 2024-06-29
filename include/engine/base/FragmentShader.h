#pragma once
#include "engine/base/VaryingStore.h"

#define ifritSampleTex(id,x)

namespace Ifrit::Engine {
	class FragmentShader {
	public:
		float* atTexture[32];
		uint32_t* atTextureWid[32];
		uint32_t* atTextureHei[32];
		IFRIT_DUAL virtual void execute(const void* varyings, void* colorOutput, int stride) =0;
		IFRIT_HOST virtual FragmentShader* getCudaClone() { return nullptr; };
	};
}