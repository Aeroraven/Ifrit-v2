#pragma once
#include "core/definition/CoreTypes.h"
namespace Ifrit::Engine {
	enum TypeDescriptorEnum {
		IFTP_UNDEFINED,
		IFTP_FLOAT1,
		IFTP_FLOAT2,
		IFTP_FLOAT3,
		IFTP_FLOAT4,
		IFTP_INT1,
		IFTP_INT2,
		IFTP_INT3,
		IFTP_INT4,
		IFTP_UINT1,
		IFTP_UINT2,
		IFTP_UINT3,
		IFTP_UINT4
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
		TypeDescriptorImpl<float, TypeDescriptorEnum::IFTP_FLOAT1> FLOAT1;
		TypeDescriptorImpl<float2, TypeDescriptorEnum::IFTP_FLOAT2> FLOAT2;
		TypeDescriptorImpl<float3, TypeDescriptorEnum::IFTP_FLOAT3> FLOAT3;
		TypeDescriptorImpl<float4, TypeDescriptorEnum::IFTP_FLOAT4> FLOAT4;
		TypeDescriptorImpl<int, TypeDescriptorEnum::IFTP_INT1> INT1;
		TypeDescriptorImpl<int2, TypeDescriptorEnum::IFTP_INT2> INT2;
		TypeDescriptorImpl<int3, TypeDescriptorEnum::IFTP_INT3> INT3;
		TypeDescriptorImpl<int4, TypeDescriptorEnum::IFTP_INT4> INT4;
		TypeDescriptorImpl<unsigned int, TypeDescriptorEnum::IFTP_UINT1> UINT1;
		TypeDescriptorImpl<uint2, TypeDescriptorEnum::IFTP_UINT2> UINT2;
		TypeDescriptorImpl<uint3, TypeDescriptorEnum::IFTP_UINT3> UINT3;
		TypeDescriptorImpl<uint4, TypeDescriptorEnum::IFTP_UINT4> UINT4;
	};

	static TypeDescriptorsT TypeDescriptors;
}