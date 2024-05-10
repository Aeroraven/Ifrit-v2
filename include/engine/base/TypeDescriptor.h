#pragma once
#include "core/definition/CoreExports.h"
namespace Ifrit::Engine {
	enum class TypeDescriptorEnum {
		UNDEFINED,
		FLOAT1,
		FLOAT2,
		FLOAT3,
		FLOAT4,
		INT1,
		INT2,
		INT3,
		INT4,
		UINT1,
		UINT2,
		UINT3,
		UINT4
	};
	struct TypeDescriptor {
		uint32_t size;
		TypeDescriptorEnum type;
	};

	template<typename T, TypeDescriptorEnum U>
	struct TypeDescriptorImpl:TypeDescriptor {
		TypeDescriptorImpl() {
			this->size = sizeof(T);
			this->type = U;
		}
	};

	class TypeDescriptorsT {
	public:
		TypeDescriptorImpl<float, TypeDescriptorEnum::FLOAT1> FLOAT1;
		TypeDescriptorImpl<float2, TypeDescriptorEnum::FLOAT2> FLOAT2;
		TypeDescriptorImpl<float3, TypeDescriptorEnum::FLOAT3> FLOAT3;
		TypeDescriptorImpl<float4, TypeDescriptorEnum::FLOAT4> FLOAT4;
		TypeDescriptorImpl<int, TypeDescriptorEnum::INT1> INT1;
		TypeDescriptorImpl<int2, TypeDescriptorEnum::INT2> INT2;
		TypeDescriptorImpl<int3, TypeDescriptorEnum::INT3> INT3;
		TypeDescriptorImpl<int4, TypeDescriptorEnum::INT4> INT4;
		TypeDescriptorImpl<unsigned int, TypeDescriptorEnum::UINT1> UINT1;
		TypeDescriptorImpl<uint2, TypeDescriptorEnum::UINT2> UINT2;
		TypeDescriptorImpl<uint3, TypeDescriptorEnum::UINT3> UINT3;
		TypeDescriptorImpl<uint4, TypeDescriptorEnum::UINT4> UINT4;
	};

	static TypeDescriptorsT TypeDescriptors;
}