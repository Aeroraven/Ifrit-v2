
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
#ifdef IFRIT_FEATURE_CUDA
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/softgraphics/core/cuda/CudaUtils.cuh"
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
#include "ifrit/softgraphics/engine/base/TypeDescriptor.h"

#include "ifrit/softgraphics/engine/base/Constants.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/base/Structures.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterCommon.h"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"

namespace Ifrit::SoftRenderer::TileRaster::CUDA::Invocation {
enum GeometryGenerationPipelineType {
  IFCUINVO_GEOMETRY_GENERATION_UNDEFINED = 0,
  IFCUINVO_GEOMETRY_GENERATION_CONVENTIONAL = 1,
  IFCUINVO_GEOMETRY_GENERATION_MESHSHADER = 2
};

struct RenderingInvocationArgumentSet {
  char *dVertexBuffer;
  TypeDescriptorEnum *dVertexTypeDescriptor;
  int *dIndexBuffer;
  VertexShader *dVertexShader;
  FragmentShader *dFragmentShader;
  GeometryShader *dGeometryShader;
  ifloat4 **dColorBuffer;
  ifloat4 **dHostColorBuffer;
  ifloat4 **hColorBuffer;
  u32 dHostColorBufferSize;
  float *dDepthBuffer;
  ifloat4 *dPositionBuffer;
  TileRasterDeviceContext *deviceContext;
  int totalIndices;
  bool doubleBuffering;
  ifloat4 **dLastColorBuffer;
  IfritPolygonMode polygonMode = IF_POLYGON_MODE_FILL;
  ifloat4 *hClearColors;
  float hClearDepth;

  MeshShader *dMeshShader;
  TaskShader *dTaskShader;
  iint3 gMeshShaderLocalSize;
  int gMeshShaderNumWorkGroups;
  GeometryGenerationPipelineType gGeometryPipelineType;
  int gMeshShaderAttributes;
};
void invokeCudaRenderingClear(const RenderingInvocationArgumentSet &args) IFRIT_AP_NOTHROW;
void invokeCudaRendering(const RenderingInvocationArgumentSet &args) IFRIT_AP_NOTHROW;

void invokeFragmentShaderUpdate(FragmentShader *dFragmentShader) IFRIT_AP_NOTHROW;
void updateFrameBufferConstants(u32 width, u32 height);
void updateScissorTestData(const ifloat4 *scissorAreas, int numScissors, bool scissorEnable);
void initCudaRendering();
void updateVertexLayout(TypeDescriptorEnum *dVertexTypeDescriptor, int attrCounts);

int *getIndexBufferDeviceAddr(const int *hIndexBuffer, u32 indexBufferSize, int *dOldIndexBuffer);
char *getVertexBufferDeviceAddr(const char *hVertexBuffer, u32 bufferSize, char *dOldBuffer);
TypeDescriptorEnum *getTypeDescriptorDeviceAddr(const TypeDescriptorEnum *hBuffer, u32 bufferSize,
                                                TypeDescriptorEnum *dOldBuffer);
float *getDepthBufferDeviceAddr(u32 bufferSize, float *dOldBuffer);
ifloat4 *getPositionBufferDeviceAddr(u32 bufferSize, ifloat4 *dOldBuffer);
void getColorBufferDeviceAddr(const std::vector<ifloat4 *> &hColorBuffer, std::vector<ifloat4 *> &dhColorBuffer,
                              ifloat4 **&dColorBuffer, u32 bufferSize, std::vector<ifloat4 *> &dhOldColorBuffer,
                              ifloat4 **dOldBuffer);
void updateAttributes(u32 attributeCounts);
void updateVarying(u32 varyingCounts);
void updateVertexCount(u32 vertexCount);

char *deviceMalloc(u32 size);
void deviceFree(char *ptr);
void createTexture(u32 texId, const IfritImageCreateInfo &createInfo, float *data);
void createSampler(u32 slotId, const IfritSamplerT &samplerState);
void createDeviceBuffer(u32 slotId, int bufferSize);
void copyHostBufferToBuffer(const void *srcBuffer, int dstSlot, int size);

void setBlendFunc(IfritColorAttachmentBlendState blendState);
void setDepthFunc(IfritCompareOp depthFunc);
void setCullMode(IfritCullMode cullMode);
void setMsaaSampleBits(IfritSampleCountFlagBits sampleBits);
} // namespace Ifrit::SoftRenderer::TileRaster::CUDA::Invocation
#endif