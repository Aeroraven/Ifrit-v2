
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


#include "ifrit/softgraphics/engine/export/EngineCommonExport.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterRenderer.h"

using namespace Ifrit::Common::Utility;

#define IFRIT_BASENS Ifrit::GraphicsBackend::SoftGraphics
#define IFRIT_CORENS Ifrit::GraphicsBackend::SoftGraphics::Core::Data

using namespace Ifrit::GraphicsBackend::SoftGraphics;
using namespace Ifrit::GraphicsBackend::SoftGraphics::Core::Data;

IFRIT_APIDECL_COMPAT IFRIT_BASENS::FrameBuffer *IFRIT_APICALL
iftrCreateFrameBuffer() IFRIT_EXPORT_COMPAT_NOTHROW {
  return new FrameBuffer();
}

IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyFrameBuffer(
    IFRIT_BASENS::FrameBuffer *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
  delete pInstance;
}

IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetFrameBufferColorAttachmentFP32(
    IFRIT_BASENS::FrameBuffer *pInstance, IFRIT_CORENS::ImageF32 *const *pImage,
    size_t nums) IFRIT_EXPORT_COMPAT_NOTHROW {
  pInstance->setColorAttachmentsCompatible(pImage, size_cast<int>(nums));
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetFrameBufferDepthAttachmentFP32(
    IFRIT_BASENS::FrameBuffer *pInstance,
    IFRIT_CORENS::ImageF32 *pImage) IFRIT_EXPORT_COMPAT_NOTHROW {
  pInstance->setDepthAttachmentCompatible(pImage);
}

IFRIT_APIDECL_COMPAT IFRIT_BASENS::VaryingDescriptor *IFRIT_APICALL
iftrCreateVaryingDescriptor() IFRIT_EXPORT_COMPAT_NOTHROW {
  return new VaryingDescriptor();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyVaryingDescriptor(
    IFRIT_BASENS::VaryingDescriptor *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
  delete pInstance;
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrWriteVaryingDescriptor(
    IFRIT_BASENS::VaryingDescriptor *pInstance,
    const Ifrit::GraphicsBackend::SoftGraphics::LibraryExport::ExportTypeDesc
        *pDesc,
    size_t num) IFRIT_EXPORT_COMPAT_NOTHROW {
  std::vector<TypeDescriptor> desc;
  for (int i = 0; i < num; i++) {
    if (pDesc[i] == TypeDescriptorEnum::IFTP_UNDEFINED)
      desc.push_back(TypeDescriptors.FLOAT4);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_FLOAT1)
      desc.push_back(TypeDescriptors.FLOAT1);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_FLOAT2)
      desc.push_back(TypeDescriptors.FLOAT2);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_FLOAT3)
      desc.push_back(TypeDescriptors.FLOAT3);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_FLOAT4)
      desc.push_back(TypeDescriptors.FLOAT4);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_INT1)
      desc.push_back(TypeDescriptors.INT1);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_INT2)
      desc.push_back(TypeDescriptors.INT2);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_INT3)
      desc.push_back(TypeDescriptors.INT3);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_INT4)
      desc.push_back(TypeDescriptors.INT4);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_UINT1)
      desc.push_back(TypeDescriptors.UINT1);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_UINT2)
      desc.push_back(TypeDescriptors.UINT2);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_UINT3)
      desc.push_back(TypeDescriptors.UINT3);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_UINT4)
      desc.push_back(TypeDescriptors.UINT4);
  }
  pInstance->setVaryingDescriptors(desc);
}

IFRIT_APIDECL_COMPAT IFRIT_BASENS::VertexBuffer *IFRIT_APICALL
iftrCreateVertexBuffer() IFRIT_EXPORT_COMPAT_NOTHROW {
  return new VertexBuffer();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyVertexBuffer(
    IFRIT_BASENS::VertexBuffer *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
  delete pInstance;
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetVertexBufferLayout(
    IFRIT_BASENS::VertexBuffer *pInstance,
    const Ifrit::GraphicsBackend::SoftGraphics::LibraryExport::ExportTypeDesc
        *pDesc,
    size_t num) IFRIT_EXPORT_COMPAT_NOTHROW {
  std::vector<TypeDescriptor> desc;
  for (int i = 0; i < num; i++) {
    if (pDesc[i] == TypeDescriptorEnum::IFTP_UNDEFINED)
      desc.push_back(TypeDescriptors.FLOAT4);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_FLOAT1)
      desc.push_back(TypeDescriptors.FLOAT1);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_FLOAT2)
      desc.push_back(TypeDescriptors.FLOAT2);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_FLOAT3)
      desc.push_back(TypeDescriptors.FLOAT3);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_FLOAT4)
      desc.push_back(TypeDescriptors.FLOAT4);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_INT1)
      desc.push_back(TypeDescriptors.INT1);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_INT2)
      desc.push_back(TypeDescriptors.INT2);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_INT3)
      desc.push_back(TypeDescriptors.INT3);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_INT4)
      desc.push_back(TypeDescriptors.INT4);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_UINT1)
      desc.push_back(TypeDescriptors.UINT1);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_UINT2)
      desc.push_back(TypeDescriptors.UINT2);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_UINT3)
      desc.push_back(TypeDescriptors.UINT3);
    else if (pDesc[i] == TypeDescriptorEnum::IFTP_UINT4)
      desc.push_back(TypeDescriptors.UINT4);
    else {
      printf("Unknown Descriptor %d \n", pDesc[i]);
    }
  }
  pInstance->setLayout(desc);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
iftrSetVertexBufferSize(IFRIT_BASENS::VertexBuffer *pInstance,
                        size_t num) IFRIT_EXPORT_COMPAT_NOTHROW {
  pInstance->setVertexCount(static_cast<int>(num));
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrAllocateVertexBuffer(
    IFRIT_BASENS::VertexBuffer *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
  pInstance->allocateBuffer(pInstance->getVertexCount());
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetVertexBufferValueFloat4(
    IFRIT_BASENS::VertexBuffer *pInstance, int index, int attr,
    void *value) IFRIT_EXPORT_COMPAT_NOTHROW {
  auto fvalue = static_cast<ifloat4 *>(value);
  pInstance->setValueFloat4Compatible(index, attr, *fvalue);
}

#undef IFRIT_CORENS
#undef IFRIT_BASENS