
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

namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster {

struct AssembledTriangleRef {
  int sourcePrimitive;
  int vertexReferences[3];
};

constexpr auto tagbufferSizeX = TileRasterContext::tileWidth;

struct TagBufferContextVec2 {
  float x, y;
};

struct TagBufferContext {
  Ifrit::Math::SIMD::vfloat3 tagBufferBary[tagbufferSizeX * tagbufferSizeX];
  Ifrit::Math::SIMD::vfloat3 atpBx[tagbufferSizeX * tagbufferSizeX];
  Ifrit::Math::SIMD::vfloat3 atpBy[tagbufferSizeX * tagbufferSizeX];
  Ifrit::Math::SIMD::vfloat4 atpF1F2[tagbufferSizeX * tagbufferSizeX];
  TagBufferContextVec2 atpF3[tagbufferSizeX * tagbufferSizeX];
  int valid[tagbufferSizeX * tagbufferSizeX];
};

struct PixelShadingFuncArgs {
  ImageF32 *depthAttachmentPtr;
  int varyingCounts;
  ImageF32 *colorAttachment0;
  const int *indexBufferPtr;
  TagBufferContext *tagBuffer;
  bool forcedInQuads;
};

class TileRasterWorker {
protected:
  std::atomic<TileRasterStage> status;

private:
  uint32_t workerId;
  TileRasterRenderer *rendererReference;
  std::unique_ptr<std::thread> execWorker;
  std::shared_ptr<TileRasterContext> context;

  std::vector<Ifrit::Math::SIMD::vfloat4> interpolatedVaryings;
  std::vector<const void *> interpolatedVaryingsAddr;

  std::vector<ifloat4> colorOutput = std::vector<ifloat4>(1);
  std::vector<AssembledTriangleProposal> generatedTriangle;

  // Hold Cache
  float depthCache[TileRasterContext::tileWidth * TileRasterContext::tileWidth];
  std::vector<TileBinProposal> coverQueueLocal;
  float curTileHierZ = 0;

  // Debug
  int totalDraws = 0;
  int reqDraws = 0;
  const float EPS = 1e-8f;
  const float EPS2 = 1e-8f;

  bool vert = false;
  bool geom = false;
  bool rast = false;
  bool frag = false;

public:
  TileRasterWorker(uint32_t workerId, TileRasterRenderer *renderer,
                   std::shared_ptr<TileRasterContext> context);

protected:
  friend class TileRasterRenderer;
  void run() IFRIT_AP_NOTHROW;
  void drawCall(bool withClear) IFRIT_AP_NOTHROW;
  void drawCallWithClear() IFRIT_AP_NOTHROW;
  void release();

  bool triangleFrustumClip(Ifrit::Math::SIMD::vfloat4 v1,
                           Ifrit::Math::SIMD::vfloat4 v2,
                           Ifrit::Math::SIMD::vfloat4 v3,
                           Ifrit::Math::SIMD::vfloat4 &bbox) IFRIT_AP_NOTHROW;
  uint32_t
  triangleHomogeneousClip(const int primitiveId, Ifrit::Math::SIMD::vfloat4 v1,
                          Ifrit::Math::SIMD::vfloat4 v2,
                          Ifrit::Math::SIMD::vfloat4 v3) IFRIT_AP_NOTHROW;
  bool triangleCulling(Ifrit::Math::SIMD::vfloat4 v1,
                       Ifrit::Math::SIMD::vfloat4 v2,
                       Ifrit::Math::SIMD::vfloat4 v3) IFRIT_AP_NOTHROW;
  void executeBinner(const int primitiveId,
                     const AssembledTriangleProposalRasterStage &atp,
                     Ifrit::Math::SIMD::vfloat4 bbox) IFRIT_AP_NOTHROW;

  void vertexProcessing(TileRasterRenderer *renderer) IFRIT_AP_NOTHROW;
  void geometryProcessing(TileRasterRenderer *renderer) IFRIT_AP_NOTHROW;
  void tiledProcessing(TileRasterRenderer *renderer,
                       bool clearDepth) IFRIT_AP_NOTHROW;

  void rasterizationSingleTile(TileRasterRenderer *renderer,
                               int tileId) IFRIT_AP_NOTHROW;
  void sortOrderProcessingSingleTile(TileRasterRenderer *renderer,
                                     int tileId) IFRIT_AP_NOTHROW;
  void fragmentProcessingSingleTile(TileRasterRenderer *renderer,
                                    bool clearedDepth,
                                    int tileId) IFRIT_AP_NOTHROW;

  void threadStart();

  template <bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc,
            bool tpOnlyTaggingPass>
  void pixelShading(const AssembledTriangleProposalShadeStage &atp,
                    const int dx, const int dy,
                    const PixelShadingFuncArgs &args) IFRIT_AP_NOTHROW;

  template <bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc,
            bool tpOnlyTaggingPass>
  void
  pixelShadingSingleQuad(const AssembledTriangleProposalShadeStage &atp,
                         int quadMask, const int dx, const int dy,
                         const PixelShadingFuncArgs &args) IFRIT_AP_NOTHROW;

  template <bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc,
            bool tpOnlyTaggingPass>
  void
  pixelShadingSIMD256Grouped(const AssembledTriangleProposalShadeStage &atp,
                             int groupsX, int groupsY, const int dx,
                             const int dy,
                             const PixelShadingFuncArgs &args) IFRIT_AP_NOTHROW;

  template <bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc,
            bool tpOnlyTaggingPass>
  void pixelShadingSIMD128(const AssembledTriangleProposalShadeStage &atp,
                           const int dx, const int dy,
                           const PixelShadingFuncArgs &args) IFRIT_AP_NOTHROW;

  template <bool tpAlphaBlendEnable, IfritCompareOp tpDepthFunc,
            bool tpOnlyTaggingPass>
  void pixelShadingSIMD256(const AssembledTriangleProposalShadeStage &atp,
                           const int dx, const int dy,
                           const PixelShadingFuncArgs &args) IFRIT_AP_NOTHROW;

  void
  pixelShadingFromTagBuffer(const int dx, const int dy,
                            const PixelShadingFuncArgs &args) IFRIT_AP_NOTHROW;
  void pixelShadingFromTagBufferQuadInvo(const int dx, const int dy,
                                         const PixelShadingFuncArgs &args)
      IFRIT_AP_NOTHROW;

  inline int getTileID(int x, int y) IFRIT_AP_NOTHROW {
    return (unsigned)y * (unsigned)context->numTilesX + (unsigned)x;
  }
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster