#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/base/FrameBuffer.h"
#include "./engine/base/VaryingDescriptor.h"

#define IFRIT_BASENS Ifrit::Engine::SoftRenderer
#define IFRIT_CORENS Ifrit::Engine::SoftRenderer::Core::Data

namespace Ifrit::Engine::SoftRenderer::LibraryExport {
using ExportTypeDesc = int;
}

/* Exporting FrameBuffer.h */
IFRIT_APIDECL_COMPAT IFRIT_BASENS::FrameBuffer *IFRIT_APICALL
iftrCreateFrameBuffer() IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyFrameBuffer(
    IFRIT_BASENS::FrameBuffer *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetFrameBufferColorAttachmentFP32(
    IFRIT_BASENS::FrameBuffer *pInstance, IFRIT_CORENS::ImageF32 *const *pImage,
    size_t nums) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetFrameBufferDepthAttachmentFP32(
    IFRIT_BASENS::FrameBuffer *pInstance,
    IFRIT_CORENS::ImageF32 *pImage) IFRIT_EXPORT_COMPAT_NOTHROW;

/* Exporting VaryingDescriptors.h */
IFRIT_APIDECL_COMPAT IFRIT_BASENS::VaryingDescriptor *IFRIT_APICALL
iftrCreateVaryingDescriptor() IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyVaryingDescriptor(
    IFRIT_BASENS::VaryingDescriptor *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrWriteVaryingDescriptor(
    IFRIT_BASENS::VaryingDescriptor *pInstance,
    const Ifrit::Engine::SoftRenderer::LibraryExport::ExportTypeDesc *pDesc,
    size_t num) IFRIT_EXPORT_COMPAT_NOTHROW;

/* Exporting VertexBuffer.h */
IFRIT_APIDECL_COMPAT IFRIT_BASENS::VertexBuffer *IFRIT_APICALL
iftrCreateVertexBuffer() IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyVertexBuffer(
    IFRIT_BASENS::VertexBuffer *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetVertexBufferLayout(
    IFRIT_BASENS::VertexBuffer *pInstance,
    const Ifrit::Engine::SoftRenderer::LibraryExport::ExportTypeDesc *pDesc,
    size_t num) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
iftrSetVertexBufferSize(IFRIT_BASENS::VertexBuffer *pInstance,
                        size_t num) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrAllocateVertexBuffer(
    IFRIT_BASENS::VertexBuffer *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetVertexBufferValueFloat4(
    IFRIT_BASENS::VertexBuffer *pInstance, int index, int attr,
    void *value) IFRIT_EXPORT_COMPAT_NOTHROW;

#undef IFRIT_CORENS
#undef IFRIT_BASENS