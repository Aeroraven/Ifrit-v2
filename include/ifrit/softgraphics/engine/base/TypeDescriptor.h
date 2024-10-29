#pragma once
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/softgraphics/core/definition/CoreTypes.h"

namespace Ifrit::GraphicsBackend::SoftGraphics {
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
struct IFRIT_APIDECL TypeDescriptor {
  uint32_t size;
  TypeDescriptorEnum type;
};

template <typename T, TypeDescriptorEnum U>
struct IFRIT_APIDECL TypeDescriptorImpl : TypeDescriptor {
  TypeDescriptorImpl() {
    this->size = sizeof(T);
    this->type = U;
  }
};

class IFRIT_APIDECL TypeDescriptorsT {
public:
  TypeDescriptorImpl<float, TypeDescriptorEnum::IFTP_FLOAT1> FLOAT1;
  TypeDescriptorImpl<ifloat2, TypeDescriptorEnum::IFTP_FLOAT2> FLOAT2;
  TypeDescriptorImpl<ifloat3, TypeDescriptorEnum::IFTP_FLOAT3> FLOAT3;
  TypeDescriptorImpl<ifloat4, TypeDescriptorEnum::IFTP_FLOAT4> FLOAT4;
  TypeDescriptorImpl<int, TypeDescriptorEnum::IFTP_INT1> INT1;
  TypeDescriptorImpl<iint2, TypeDescriptorEnum::IFTP_INT2> INT2;
  TypeDescriptorImpl<iint3, TypeDescriptorEnum::IFTP_INT3> INT3;
  TypeDescriptorImpl<iint4, TypeDescriptorEnum::IFTP_INT4> INT4;
  TypeDescriptorImpl<unsigned int, TypeDescriptorEnum::IFTP_UINT1> UINT1;
  TypeDescriptorImpl<iuint2, TypeDescriptorEnum::IFTP_UINT2> UINT2;
  TypeDescriptorImpl<iuint3, TypeDescriptorEnum::IFTP_UINT3> UINT3;
  TypeDescriptorImpl<iuint4, TypeDescriptorEnum::IFTP_UINT4> UINT4;
};

IFRIT_APIDECL extern TypeDescriptorsT TypeDescriptors;
} // namespace Ifrit::GraphicsBackend::SoftGraphics