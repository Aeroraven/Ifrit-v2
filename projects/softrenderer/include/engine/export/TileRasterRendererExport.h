#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/tileraster/TileRasterRenderer.h"

#define IFRIT_TRNS Ifrit::Engine::TileRaster
#define IFRIT_BASENS Ifrit::Engine
#define IFRIT_TRTP Ifrit::Engine::LibraryExport::TileRasterRendererWrapper

namespace Ifrit::Engine::LibraryExport {
struct TileRasterRendererWrapper;
}

IFRIT_APIDECL_COMPAT IFRIT_TRTP *IFRIT_APICALL iftrCreateInstance()
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
iftrDestroyInstance(IFRIT_TRTP *hInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFrameBuffer(
    IFRIT_TRTP *hInstance,
    IFRIT_BASENS::FrameBuffer *frameBuffer) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexBuffer(
    IFRIT_TRTP *hInstance,
    const IFRIT_BASENS::VertexBuffer *vertexBuffer) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindIndexBuffer(
    IFRIT_TRTP *hInstance, void *indexBuffer) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexShaderFunc(
    IFRIT_TRTP *hInstance, IFRIT_BASENS::VertexShaderFunctionalPtr func,
    IFRIT_BASENS::VaryingDescriptor *vsOutDescriptors)
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFragmentShaderFunc(
    IFRIT_TRTP *hInstance,
    IFRIT_BASENS::FragmentShaderFunctionalPtr func) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetBlendFunc(
    IFRIT_TRTP *hInstance, IFRIT_BASENS::IfritColorAttachmentBlendState *state)
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetDepthFunc(
    IFRIT_TRTP *hInstance,
    IFRIT_BASENS::IfritCompareOp state) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrOptsetForceDeterministic(
    IFRIT_TRTP *hInstance, int opt) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrOptsetDepthTestEnable(
    IFRIT_TRTP *hInstance, int opt) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
iftrDrawLegacy(IFRIT_TRTP *hInstance, int numVertices,
               int clearFramebuffer) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrClear(IFRIT_TRTP *hInstance)
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrInit(IFRIT_TRTP *hInstance)
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrTest(void (*p)(int))
    IFRIT_EXPORT_COMPAT_NOTHROW;

// Update v1
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexShader(
    IFRIT_TRTP *hInstance, void *pVertexShader) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFragmentShader(
    IFRIT_TRTP *hInstance, void *pFragmentShader) IFRIT_EXPORT_COMPAT_NOTHROW;

#undef IFRIT_TRTP
#undef IFRIT_BASENS
#undef IFRIT_TRNS