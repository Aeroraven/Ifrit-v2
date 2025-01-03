
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


#ifdef IFRIT_FEATURE_CUDA
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterRendererCuda.h"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterConstantsCuda.h"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterImageOpInvocationsCuda.cuh"
namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA {
IFRIT_APIDECL void TileRasterRendererCuda::init() {
  context = std::make_unique<TileRasterContextCuda>();
  deviceContext = std::make_unique<TileRasterDeviceContext>();

  deviceContext->dShadingQueue =
      (uint32_t *)Invocation::deviceMalloc(sizeof(uint32_t));
  deviceContext->dDeviceConstants =
      (TileRasterDeviceConstants *)Invocation::deviceMalloc(
          sizeof(TileRasterDeviceConstants));
  Invocation::initCudaRendering();
  context->geometryShader = nullptr;
  context->blendState.blendEnable = false;
  Invocation::setBlendFunc(context->blendState);
  Invocation::updateScissorTestData(context->scissorAreas.data(),
                                    context->scissorAreas.size(),
                                    context->scissorTestEnable);
}
IFRIT_APIDECL void
TileRasterRendererCuda::bindFrameBuffer(FrameBuffer &frameBuffer,
                                        bool useDoubleBuffer) {
  context->frameBuffer = &frameBuffer;
  auto pixelCount = frameBuffer.getWidth() * frameBuffer.getHeight();
  this->deviceDepthBuffer =
      Invocation::getDepthBufferDeviceAddr(pixelCount, this->deviceDepthBuffer);

  std::vector<ifloat4 *> hColorBuffer = {
      (ifloat4 *)frameBuffer.getColorAttachment(0)->getData()};

  Invocation::getColorBufferDeviceAddr(
      hColorBuffer, this->deviceHostColorBuffers[0], this->deviceColorBuffer[0],
      pixelCount, this->deviceHostColorBuffers[0], this->deviceColorBuffer[0]);
  Invocation::getColorBufferDeviceAddr(
      hColorBuffer, this->deviceHostColorBuffers[1], this->deviceColorBuffer[1],
      pixelCount, this->deviceHostColorBuffers[1], this->deviceColorBuffer[1]);
  this->doubleBuffer = useDoubleBuffer;
  this->hostColorBuffers = hColorBuffer;
  Invocation::updateFrameBufferConstants(frameBuffer.getWidth(),
                                         frameBuffer.getHeight());
}
IFRIT_APIDECL void
TileRasterRendererCuda::bindVertexBuffer(const VertexBuffer &vertexBuffer) {
  needVaryingUpdate = true;
  context->vertexBuffer = &vertexBuffer;
  char *hVertexBuffer = context->vertexBuffer->getBufferUnsafe();
  uint32_t hVertexBufferSize = context->vertexBuffer->getBufferSize();
  this->deviceVertexBuffer = Invocation::getVertexBufferDeviceAddr(
      hVertexBuffer, hVertexBufferSize, this->deviceVertexBuffer);
  this->devicePosBuffer = Invocation::getPositionBufferDeviceAddr(
      hVertexBufferSize, this->devicePosBuffer);

  std::vector<TypeDescriptorEnum> hVertexBufferLayout;
  for (int i = 0; i < context->vertexBuffer->getAttributeCount(); i++) {
    hVertexBufferLayout.push_back(
        context->vertexBuffer->getAttributeDescriptor(i).type);
  }
  this->deviceVertexTypeDescriptor = Invocation::getTypeDescriptorDeviceAddr(
      hVertexBufferLayout.data(), hVertexBufferLayout.size(),
      this->deviceVertexTypeDescriptor);

  Invocation::updateVertexLayout(hVertexBufferLayout.data(),
                                 hVertexBufferLayout.size());
  Invocation::updateVertexCount(vertexBuffer.getVertexCount());
  Invocation::updateAttributes(vertexBuffer.getAttributeCount());
}
IFRIT_APIDECL void
TileRasterRendererCuda::bindIndexBuffer(const std::vector<int> &indexBuffer) {
  context->indexBuffer = &indexBuffer;
  this->deviceIndexBuffer = Invocation::getIndexBufferDeviceAddr(
      indexBuffer.data(), indexBuffer.size(), this->deviceIndexBuffer);
}

IFRIT_APIDECL void
TileRasterRendererCuda::bindVertexShader(VertexShader *vertexShader,
                                         VaryingDescriptor &varyingDescriptor) {
  context->vertexShader = vertexShader;
  context->varyingDescriptor = &varyingDescriptor;
  std::vector<TypeDescriptorEnum> hVaryingBufferLayout;
  for (int i = 0; i < context->varyingDescriptor->getVaryingCounts(); i++) {
    hVaryingBufferLayout.push_back(
        context->varyingDescriptor->getVaryingDescriptor(i).type);
  }
  this->deviceVaryingTypeDescriptor = Invocation::getTypeDescriptorDeviceAddr(
      hVaryingBufferLayout.data(), hVaryingBufferLayout.size(),
      this->deviceVaryingTypeDescriptor);

  needVaryingUpdate = true;
  Invocation::updateVarying(context->varyingDescriptor->getVaryingCounts());
}
void TileRasterRendererCuda::updateVaryingBuffer() {
  if (!needVaryingUpdate)
    return;
  needVaryingUpdate = false;
  if (context->vertexBuffer == nullptr)
    return;
  if (context->varyingDescriptor == nullptr)
    return;

  auto vcount = context->varyingDescriptor->getVaryingCounts();
  auto vxcount = context->vertexBuffer->getVertexCount();
  cudaMalloc(&deviceContext->dVaryingBufferM2,
             vcount * vxcount * sizeof(VaryingStore));
}
IFRIT_APIDECL void
TileRasterRendererCuda::bindFragmentShader(FragmentShader *fragmentShader) {
  context->fragmentShader = fragmentShader;
  needFragmentShaderUpdate = true;
  Invocation::setBlendFunc(context->blendState);
}
IFRIT_APIDECL void
TileRasterRendererCuda::bindGeometryShader(GeometryShader *geometryShader) {
  context->geometryShader = geometryShader;
}
IFRIT_APIDECL void
TileRasterRendererCuda::bindMeshShader(MeshShader *meshShader,
                                       VaryingDescriptor &varyingDescriptor,
                                       iint3 localSize) {
  context->meshShader = meshShader;
  context->meshShaderAttributCnt = varyingDescriptor.getVaryingCounts();
  if (context->taskShader == nullptr) {
    context->meshShaderBlockSize = localSize;
  }
}
IFRIT_APIDECL void
TileRasterRendererCuda::bindTaskShader(TaskShader *taskShader,
                                       VaryingDescriptor &varyingDescriptor) {
  context->taskShader = taskShader;
  context->meshShaderAttributCnt = varyingDescriptor.getVaryingCounts();
}
IFRIT_APIDECL void
TileRasterRendererCuda::createTexture(int slotId,
                                      const IfritImageCreateInfo &createInfo) {
  Invocation::createTexture(slotId, createInfo, nullptr);
  needFragmentShaderUpdate = true;
}
IFRIT_APIDECL void
TileRasterRendererCuda::createSampler(int slotId,
                                      const IfritSamplerT &samplerState) {
  Invocation::createSampler(slotId, samplerState);
  needFragmentShaderUpdate = true;
}
IFRIT_APIDECL void TileRasterRendererCuda::createBuffer(int slotId,
                                                        int bufSize) {
  Invocation::createDeviceBuffer(slotId, bufSize);
}
IFRIT_APIDECL void
TileRasterRendererCuda::setRasterizerPolygonMode(IfritPolygonMode mode) {
  this->polygonMode = mode;
}
IFRIT_APIDECL void
TileRasterRendererCuda::setBlendFunc(IfritColorAttachmentBlendState state) {
  context->blendState = state;
  Invocation::setBlendFunc(state);
}
IFRIT_APIDECL void
TileRasterRendererCuda::setDepthFunc(IfritCompareOp depthFunc) {
  ctxDepthFunc = depthFunc;
  if (ctxDepthTestEnable) {
    Invocation::setDepthFunc(depthFunc);
  }
}
IFRIT_APIDECL void TileRasterRendererCuda::setDepthTestEnable(bool option) {
  ctxDepthTestEnable = option;
  if (option) {
    Invocation::setDepthFunc(ctxDepthFunc);
  } else {
    Invocation::setDepthFunc(IF_COMPARE_OP_ALWAYS);
  }
}
IFRIT_APIDECL void
TileRasterRendererCuda::setScissors(const std::vector<ifloat4> &scissors) {
  context->scissorAreas = scissors;
  Invocation::updateScissorTestData(scissors.data(), scissors.size(),
                                    context->scissorTestEnable);
}
IFRIT_APIDECL void TileRasterRendererCuda::setScissorTestEnable(bool option) {
  context->scissorTestEnable = option;
  Invocation::updateScissorTestData(context->scissorAreas.data(),
                                    context->scissorAreas.size(),
                                    context->scissorTestEnable);
}
IFRIT_APIDECL void TileRasterRendererCuda::setCullMode(IfritCullMode cullMode) {
  Invocation::setCullMode(cullMode);
}
IFRIT_APIDECL void
TileRasterRendererCuda::setMsaaSamples(IfritSampleCountFlagBits msaaSamples) {
  Invocation::setMsaaSampleBits(msaaSamples);
}
IFRIT_APIDECL void
TileRasterRendererCuda::setClearValues(const std::vector<ifloat4> &clearColors,
                                       float clearDepth) {
  ctxClearColors = clearColors;
  ctxClearDepth = clearDepth;
}
IFRIT_APIDECL void TileRasterRendererCuda::clear() {
  // ifloat4* colorBuffer =
  // (ifloat4*)context->frameBuffer->getColorAttachment(0)->getData();
  int totalIndices = 0;
  if (totalIndices)
    totalIndices = context->indexBuffer->size();
  int curBuffer = currentBuffer;
  Invocation::RenderingInvocationArgumentSet args;
  args.hClearColors = ctxClearColors.data();
  args.hClearDepth = ctxClearDepth;
  args.dVertexBuffer = deviceVertexBuffer;
  args.dVertexTypeDescriptor = deviceVertexTypeDescriptor;
  args.dIndexBuffer = deviceIndexBuffer;
  args.dVertexShader = context->vertexShader;
  args.dFragmentShader = context->fragmentShader;
  args.dGeometryShader = context->geometryShader;
  args.dColorBuffer = deviceColorBuffer[curBuffer];
  args.dHostColorBuffer = deviceHostColorBuffers[curBuffer].data();
  args.hColorBuffer = hostColorBuffers.data();
  args.dHostColorBufferSize = deviceHostColorBuffers[curBuffer].size();
  args.dDepthBuffer = deviceDepthBuffer;
  args.dPositionBuffer = devicePosBuffer;
  args.deviceContext = this->deviceContext.get();
  args.totalIndices = totalIndices;
  args.doubleBuffering = this->doubleBuffer;
  args.dLastColorBuffer = deviceHostColorBuffers[1 - curBuffer].data();
  args.polygonMode = polygonMode;
  Invocation::invokeCudaRenderingClear(args);
}
void TileRasterRendererCuda::initCuda() {
  if (this->initCudaContext)
    return;
  this->initCudaContext = true;
  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 8192);
}
IFRIT_APIDECL void TileRasterRendererCuda::generateMipmap(int slotId,
                                                          IfritFilter filter) {
  Invocation::invokeMipmapGeneration(slotId, filter);
}
IFRIT_APIDECL void
TileRasterRendererCuda::blitImage(int srcSlotId, int dstSlotId,
                                  const IfritImageBlit &region,
                                  IfritFilter filter) {
  Invocation::invokeBlitImage(srcSlotId, dstSlotId, region, filter);
}
IFRIT_APIDECL void TileRasterRendererCuda::copyHostBufferToImage(
    void *srcBuffer, int dstSlot,
    const std::vector<IfritBufferImageCopy> &regions) {
  Invocation::invokeCopyBufferToImage(srcBuffer, dstSlot, regions.size(),
                                      regions.data());
}
IFRIT_APIDECL void
TileRasterRendererCuda::copyHostBufferToBuffer(const void *srcBuffer,
                                               int dstSlot, int size) {
  Invocation::copyHostBufferToBuffer(srcBuffer, dstSlot, size);
}
void TileRasterRendererCuda::internalRender(
    TileRasterRendererCudaVertexPipelineType vertexPipeType) {
  initCuda();
  updateVaryingBuffer();

  if (needFragmentShaderUpdate) {
    Invocation::invokeFragmentShaderUpdate(context->fragmentShader);
    needFragmentShaderUpdate = false;
  }

  // ifloat4* colorBuffer =
  // (ifloat4*)context->frameBuffer->getColorAttachment(0)->getData();
  int totalIndices = 0;
  if (context->indexBuffer)
    totalIndices = context->indexBuffer->size();
  int curBuffer = currentBuffer;
  Invocation::RenderingInvocationArgumentSet args;
  args.dVertexBuffer = deviceVertexBuffer;
  args.dVertexTypeDescriptor = deviceVertexTypeDescriptor;
  args.dIndexBuffer = deviceIndexBuffer;
  args.dVertexShader = context->vertexShader;
  args.dFragmentShader = context->fragmentShader;
  args.dGeometryShader = context->geometryShader;
  args.dColorBuffer = deviceColorBuffer[curBuffer];
  args.dHostColorBuffer = deviceHostColorBuffers[curBuffer].data();
  args.hColorBuffer = hostColorBuffers.data();
  args.dHostColorBufferSize = deviceHostColorBuffers[curBuffer].size();
  args.dDepthBuffer = deviceDepthBuffer;
  args.dPositionBuffer = devicePosBuffer;
  args.deviceContext = this->deviceContext.get();
  args.totalIndices = totalIndices;
  args.doubleBuffering = this->doubleBuffer;
  args.dLastColorBuffer = deviceHostColorBuffers[1 - curBuffer].data();
  args.polygonMode = polygonMode;

  if (vertexPipeType == IFINTERNAL_CU_VERTEX_PIPELINE_CONVENTIONAL) {
    args.gGeometryPipelineType =
        Invocation::IFCUINVO_GEOMETRY_GENERATION_CONVENTIONAL;
  } else if (vertexPipeType == IFINTERNAL_CU_VERTEX_PIPELINE_MESHSHADER) {
    args.gGeometryPipelineType =
        Invocation::IFCUINVO_GEOMETRY_GENERATION_MESHSHADER;
    args.dMeshShader = context->meshShader;
    args.gMeshShaderLocalSize = context->meshShaderBlockSize;
    args.gMeshShaderNumWorkGroups = context->meshShaderNumWorkGroups;
    args.gMeshShaderAttributes = context->meshShaderAttributCnt;
    args.dTaskShader = context->taskShader;
  }

  Invocation::invokeCudaRendering(args);
  currentBuffer = 1 - curBuffer;
}

IFRIT_APIDECL void TileRasterRendererCuda::drawElements() {
  internalRender(IFINTERNAL_CU_VERTEX_PIPELINE_CONVENTIONAL);
}
IFRIT_APIDECL void TileRasterRendererCuda::drawMeshTasks(int numWorkGroups,
                                                         int firstWorkGroup) {
  ifritAssert(firstWorkGroup == 0, "workgroup offset not supported now");
  context->meshShaderNumWorkGroups = numWorkGroups;
  internalRender(IFINTERNAL_CU_VERTEX_PIPELINE_MESHSHADER);
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA
#endif