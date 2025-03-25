
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

#if IFRIT_USE_CUDA
#include "ifrit/softgraphics/engine/base/Constants.h"
#include "ifrit/softgraphics/engine/base/VaryingDescriptor.h"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterContextCuda.h"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"

namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA {
using namespace Ifrit::GraphicsBackend::SoftGraphics;

class TileRasterRendererCuda : public std::enable_shared_from_this<TileRasterRendererCuda> {
private:
  std::unique_ptr<TileRasterContextCuda> context;
  std::unique_ptr<TileRasterDeviceContext> deviceContext;
  bool needVaryingUpdate = true;
  bool needFragmentShaderUpdate = true;
  bool initCudaContext = false;

  // Depth Test
  IfritCompareOp ctxDepthFunc = IF_COMPARE_OP_LESS;
  bool ctxDepthTestEnable = true;
  std::vector<Vector4f> ctxClearColors = {{0.0f, 0.0f, 0.0f, 0.0f}};
  float ctxClearDepth = 1.0f;

  // Device Addrs
  int *deviceIndexBuffer = nullptr;
  char *deviceVertexBuffer = nullptr;
  TypeDescriptorEnum *deviceVertexTypeDescriptor = nullptr;
  TypeDescriptorEnum *deviceVaryingTypeDescriptor = nullptr;
  float *deviceDepthBuffer = nullptr;
  Vector4f *devicePosBuffer = nullptr;
  int *deviceShadingLockBuffer = nullptr;

  std::vector<Vector4f *> deviceHostColorBuffers[2];
  Vector4f **deviceColorBuffer[2] = {nullptr, nullptr};
  std::vector<Vector4f *> hostColorBuffers{};

  bool doubleBuffer = false;
  int currentBuffer = 0;

  // Render confs
  IfritPolygonMode polygonMode = IF_POLYGON_MODE_FILL;

private:
  enum TileRasterRendererCudaVertexPipelineType {
    IFINTERNAL_CU_VERTEX_PIPELINE_UNDEFINED = 0,
    IFINTERNAL_CU_VERTEX_PIPELINE_CONVENTIONAL = 1,
    IFINTERNAL_CU_VERTEX_PIPELINE_MESHSHADER = 2
  };

private:
  void updateVaryingBuffer();
  void internalRender(TileRasterRendererCudaVertexPipelineType vertexPipeType);
  void initCuda();

public:
  IFRIT_APIDECL void init();
  IFRIT_APIDECL void bindFrameBuffer(FrameBuffer &frameBuffer, bool useDoubleBuffer = true);
  IFRIT_APIDECL void bindVertexBuffer(const VertexBuffer &vertexBuffer);
  IFRIT_APIDECL void bindIndexBuffer(const std::vector<int> &indexBuffer);
  IFRIT_APIDECL void bindVertexShader(VertexShader *vertexShader, VaryingDescriptor &varyingDescriptor);
  IFRIT_APIDECL void bindFragmentShader(FragmentShader *fragmentShader);
  IFRIT_APIDECL void bindGeometryShader(GeometryShader *geometryShader);
  IFRIT_APIDECL void bindMeshShader(MeshShader *meshShader, VaryingDescriptor &varyingDescriptor, Vector3i localSize);
  IFRIT_APIDECL void bindTaskShader(TaskShader *taskShader, VaryingDescriptor &varyingDescriptor);

  IFRIT_APIDECL void createTexture(int slotId, const IfritImageCreateInfo &createInfo);
  IFRIT_APIDECL void createSampler(int slotId, const IfritSamplerT &samplerState);
  IFRIT_APIDECL void generateMipmap(int slotId, IfritFilter filter);
  IFRIT_APIDECL void blitImage(int srcSlotId, int dstSlotId, const IfritImageBlit &region, IfritFilter filter);
  IFRIT_APIDECL void copyHostBufferToImage(void *srcBuffer, int dstSlot,
                                           const std::vector<IfritBufferImageCopy> &regions);

  IFRIT_APIDECL void createBuffer(int slotId, int bufSize);
  IFRIT_APIDECL void copyHostBufferToBuffer(const void *srcBuffer, int dstSlot, int size);

  IFRIT_APIDECL void setScissors(const std::vector<Vector4f> &scissors);
  IFRIT_APIDECL void setScissorTestEnable(bool option);

  IFRIT_APIDECL void setMsaaSamples(IfritSampleCountFlagBits msaaSamples);

  IFRIT_APIDECL void setRasterizerPolygonMode(IfritPolygonMode mode);
  IFRIT_APIDECL void setBlendFunc(IfritColorAttachmentBlendState state);
  IFRIT_APIDECL void setDepthFunc(IfritCompareOp depthFunc);
  IFRIT_APIDECL void setDepthTestEnable(bool option);
  IFRIT_APIDECL void setCullMode(IfritCullMode cullMode);
  IFRIT_APIDECL void setClearValues(const std::vector<Vector4f> &clearColors, float clearDepth);

  IFRIT_APIDECL void clear();
  IFRIT_APIDECL void drawElements();
  IFRIT_APIDECL void drawMeshTasks(int numWorkGroups, int firstWorkGroup);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA
#endif