
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
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterRenderer.h"

#define IFRIT_TRNS Ifrit::GraphicsBackend::SoftGraphics::TileRaster
#define IFRIT_BASENS Ifrit::GraphicsBackend::SoftGraphics
#define IFRIT_TRTP                                                             \
  Ifrit::GraphicsBackend::SoftGraphics::LibraryExport::TileRasterRendererWrapper

namespace Ifrit::GraphicsBackend::SoftGraphics::LibraryExport {
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