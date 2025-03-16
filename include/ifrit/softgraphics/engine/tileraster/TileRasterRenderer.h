
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
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/bufferman/BufferManager.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterContext.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster {
using namespace Ifrit::GraphicsBackend::SoftGraphics;

enum class TileRasterStage {
  IDLE,
  DRAWCALL_START,
  DRAWCALL_START_CLEAR,
  DRAWCALL_SYNC,
  VERTEX_SHADING,
  VERTEX_SHADING_SYNC,
  GEOMETRY_PROCESSING,
  GEOMETRY_PROCESSING_SYNC,
  RASTERIZATION,
  RASTERIZATION_SYNC,
  SORTING,
  SORTING_SYNC,
  FRAGMENT_SHADING,
  FRAGMENT_SHADING_SYNC,
  COMPLETED,
  TERMINATING
};
class TileRasterWorker;

class TileRasterRenderer : public Renderer, public std::enable_shared_from_this<TileRasterRenderer> {

private:
  bool shaderBindingDirtyFlag = true;
  bool varyingBufferDirtyFlag = true;
  std::shared_ptr<TileRasterContext> context;
  std::vector<std::unique_ptr<TileRasterWorker>> workers;
  std::unique_ptr<TileRasterWorker> selfOwningWorker;
  std::mutex lock;
  std::atomic<u32> unresolvedTileRaster = 0;
  std::atomic<u32> unresolvedTileFragmentShading = 0;
  std::atomic<u32> unresolvedTileSort = 0;
  std::atomic<u32> unresolvedChunkVertex = 0;
  std::atomic<u32> unresolvedChunkGeometry = 0;

  bool initialized = false;

protected:
  void createWorkers();
  void resetWorkers(TileRasterStage expectedStage);
  void statusTransitionBarrier2(TileRasterStage waitOn, TileRasterStage proceedTo);
  void statusTransitionBarrier3(TileRasterStage waitOn, TileRasterStage proceedTo);

  int fetchUnresolvedChunkVertex();
  int fetchUnresolvedChunkGeometry();

  int fetchUnresolvedTileRaster();
  int fetchUnresolvedTileFragmentShading();
  int fetchUnresolvedTileSort();

  void updateVectorCapacity();
  void updateUniformBuffer();

public:
  friend class TileRasterWorker;
  IFRIT_APIDECL TileRasterRenderer();
  IFRIT_APIDECL ~TileRasterRenderer();
  IFRIT_APIDECL void bindFrameBuffer(FrameBuffer &frameBuffer);
  IFRIT_APIDECL void bindVertexBuffer(const VertexBuffer &vertexBuffer);
  IFRIT_APIDECL void bindIndexBuffer(BufferManager::IfritBuffer indexBuffer);
  IFRIT_APIDECL void bindVertexShader(VertexShader &vertexShader);
  IFRIT_APIDECL void bindVertexShaderLegacy(VertexShader &vertexShader, VaryingDescriptor &varyingDescriptor);
  IFRIT_APIDECL void bindFragmentShader(FragmentShader &fragmentShader);
  IFRIT_APIDECL void bindUniformBuffer(int binding, int set, BufferManager::IfritBuffer pBuffer);
  IFRIT_APIDECL void setBlendFunc(IfritColorAttachmentBlendState state);
  IFRIT_APIDECL void setDepthFunc(IfritCompareOp depthFunc);

  IFRIT_APIDECL void intializeRenderContext();
  IFRIT_APIDECL void optsetForceDeterministic(bool opt);
  IFRIT_APIDECL void optsetDepthTestEnable(bool opt);

  IFRIT_APIDECL void drawElements(int vertexCount, bool clearFramebuffer) IFRIT_AP_NOTHROW;
  IFRIT_APIDECL void clear();
  IFRIT_APIDECL void init();
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster