#pragma once

#include "core/data/Image.h"
#include "engine/base/FrameBuffer.h"
#include "engine/base/Renderer.h"
#include "engine/base/Shaders.h"
#include "engine/base/VaryingDescriptor.h"
#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShaderResult.h"
#include "engine/tileraster/TileRasterCommon.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::TileRaster {
struct TileRasterContextRasterQueueProposal {
  uint32_t workerId;
  int primId;
};
struct AssembledTriangleProposalShadeStage {
  Ifrit::Math::SIMD::vfloat4 f1, f2, f3; // Interpolate Bases
  Ifrit::Math::SIMD::vfloat3 bx, by;
  int originalPrimitive;
};
struct AssembledTriangleProposalRasterStage {
  Ifrit::Math::SIMD::vfloat3 e1, e2, e3; // Edge Coefs
};

class TileRasterContext {
public:
  // Config
  constexpr static int numThreads = 16;
  constexpr static int vertexStride = 3;
  constexpr static int tileWidth = 16;
  constexpr static int subtileBlockWidth = 4;
  constexpr static int numSubtilesPerTileX = tileWidth / subtileBlockWidth;
  constexpr static int vsChunkSize = 48;
  constexpr static int gsChunkSize = 128;

  constexpr static int workerReprBits = 8;

  // Non-owning Bindings
  FrameBuffer *frameBuffer;
  const VertexBuffer *vertexBuffer;
  const int *indexBuffer;
  int indexBufferSize;
  VertexShader *vertexShader;
  VaryingDescriptor *varyingDescriptor;
  FragmentShader *fragmentShader;
  std::unordered_map<std::pair<int, int>, const void *,
                     Ifrit::Engine::GraphicsBackend::SoftGraphics::Core::Utility::PairHash>
      uniformMapping;

  // Cached attributes
  int frameWidth;
  int frameHeight;
  float invFrameWidth;
  float invFrameHeight;

  // Owning Bindings
  std::unique_ptr<VaryingDescriptor> owningVaryingDesc;

  // Thread-safe Calls
  VertexShader *threadSafeVS[TileRasterContext::numThreads + 1];
  FragmentShader *threadSafeFS[TileRasterContext::numThreads + 1];

  std::unique_ptr<VertexShader>
      threadSafeVSOwningSection[TileRasterContext::numThreads + 1];
  std::unique_ptr<FragmentShader>
      threadSafeFSOwningSection[TileRasterContext::numThreads + 1];

  // Resources
  std::unique_ptr<VertexShaderResult> vertexShaderResult;
  std::vector<std::vector<int>>
      rasterizerQueue[TileRasterContext::numThreads + 1];
  std::vector<std::vector<int>> coverQueue[TileRasterContext::numThreads + 1];

  // Sorted List
  std::vector<std::vector<TileBinProposal>> sortedCoverQueue;
  int numTilesX = 1;
  int numTilesY = 1;

  TileRasterFrontFace frontface = TileRasterFrontFace::CLOCKWISE;
  IfritColorAttachmentBlendState blendState;
  AlphaBlendingCoefs blendColorCoefs;
  AlphaBlendingCoefs blendAlphaCoefs;
  IfritCompareOp depthFunc = IF_COMPARE_OP_LESS;
  IfritCompareOp depthFuncSaved = IF_COMPARE_OP_LESS;

  // Options
  bool optForceDeterministic = true;
  bool optDepthTestEnableII = true;

  // Geometry
  std::vector<AssembledTriangleProposalRasterStage>
      assembledTrianglesRaster[TileRasterContext::numThreads + 1];
  std::vector<AssembledTriangleProposalShadeStage>
      assembledTrianglesShade[TileRasterContext::numThreads + 1];
};
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::TileRaster