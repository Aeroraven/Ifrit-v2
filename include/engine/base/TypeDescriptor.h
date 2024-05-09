#pragma once
#include "core/definition/CoreExports.h"
namespace Ifrit::Engine {

	struct TypeDescriptor {
		uint32_t size;
	};

	template<typename T>
	struct TypeDescriptorImpl:TypeDescriptor {
		using type = T;
		uint32_t size = sizeof(T);
	};

	class TypeDescriptors {
	public:
		static TypeDescriptorImpl<float> FLOAT1;
		static TypeDescriptorImpl<float2> FLOAT2;
		static TypeDescriptorImpl<float3> FLOAT3;
		static TypeDescriptorImpl<float4> FLOAT4;
		static TypeDescriptorImpl<int> INT1;
		static TypeDescriptorImpl<int2> INT2;
		static TypeDescriptorImpl<int3> INT3;
		static TypeDescriptorImpl<int4> INT4;
		static TypeDescriptorImpl<unsigned int> UINT1;
		static TypeDescriptorImpl<uint2> UINT2;
		static TypeDescriptorImpl<uint3> UINT3;
		static TypeDescriptorImpl<uint4> UINT4;

	};
}