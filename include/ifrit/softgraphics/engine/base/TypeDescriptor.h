
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/common/base/IfritBase.h"
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
  u32 size;
  TypeDescriptorEnum type;
};

template <typename T, TypeDescriptorEnum U> struct IFRIT_APIDECL TypeDescriptorImpl : TypeDescriptor {
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