
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
  Vector3i meshShaderBlockSize;
  int meshShaderAttributCnt;
  int meshShaderNumWorkGroups;

  // == scissor test ==
  bool scissorTestEnable = false;
  std::vector<Vector4f> scissorAreas;

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

  u32 **dRasterizerQueueSize = nullptr;
  u32 **dCoverQueueSize = nullptr;
  u32 *dAssembledTrianglesSize = nullptr;

  irect2Df *dTileBounds = nullptr;
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster::CUDA