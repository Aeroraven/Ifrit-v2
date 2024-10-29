#pragma once
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
#include "ifrit/softgraphics/core/definition/CoreTypes.h"

#include "ifrit/softgraphics/core/data/Image.h"
#include "ifrit/softgraphics/engine/base/FrameBuffer.h"
#include "ifrit/softgraphics/engine/base/Renderer.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/base/VaryingDescriptor.h"
#include "ifrit/softgraphics/engine/base/VertexBuffer.h"
#include "ifrit/softgraphics/engine/base/VertexShaderResult.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterCommon.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA {
using namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster;
using namespace Ifrit::GraphicsBackend::SoftGraphics;

class TileRasterContextCuda {
public:
  // Non-owning Bindings
  FrameBuffer *frameBuffer;
  const VertexBuffer *vertexBuffer = nullptr;
  const std::vector<int> *indexBuffer = nullptr;
  VertexShader *vertexShader;
  FragmentShader *fragmentShader;
  GeometryShader *geometryShader;
  VaryingDescriptor *varyingDescriptor;

  // == mesh shader ==
  MeshShader *meshShader = nullptr;
  TaskShader *taskShader = nullptr;
  iint3 meshShaderBlockSize;
  int meshShaderAttributCnt;
  int meshShaderNumWorkGroups;

  // == scissor test ==
  bool scissorTestEnable = false;
  std::vector<ifloat4> scissorAreas;

  // == blend state ==
  IfritColorAttachmentBlendState blendState;

  // Constants
  int vertexStride = 3;
  int tileBlocksX = 64;
  int subtileBlocksX = 4;
  int vertexProcessingThreads = 128;
  TileRasterDeviceConstants hostConstants;

  // CUDA Contexts
  TileRasterDeviceConstants *deviceConstants;

  char **dVertexBuffer = nullptr;
  int *dIndexBuffer = nullptr;
  char **dVaryingBuffer = nullptr;
  float **dColorBuffer = nullptr;
  float *dDepthBuffer = nullptr;

  TileBinProposal ***dRasterizerQueue = nullptr;
  TileBinProposal ***dCoverQueue = nullptr;
  AssembledTriangleProposal **dAssembledTriangles = nullptr;

  uint32_t **dRasterizerQueueSize = nullptr;
  uint32_t **dCoverQueueSize = nullptr;
  uint32_t *dAssembledTrianglesSize = nullptr;

  irect2Df *dTileBounds = nullptr;
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA