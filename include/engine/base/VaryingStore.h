#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Engine {
	union VaryingStore {
		float vf;
		int vi;
		uint32_t vui;
		float2 vf2;
		float3 vf3;
		float4 vf4;
		int2 vi2;
		int3 vi3;
		int4 vi4;
		uint2 vui2;
	};
}
