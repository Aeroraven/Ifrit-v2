#pragma once
#include "ifrit/core/renderer/postprocessing/PostFxStockhamDFT2.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/math/fastutil/FastUtil.h"
#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Core::PostprocessPassCollection {

IFRIT_APIDECL PostFxStockhamDFT2::PostFxStockhamDFT2(IApplication *app)
    : PostprocessPass(app, {"StockhamDFT2.comp.glsl", 12, 1, true}) {}

IFRIT_APIDECL void PostFxStockhamDFT2::runCommand(const GPUCmdBuffer *cmd, uint32_t wgX, uint32_t wgY, const void *pc) {
  m_computePipeline->setRecordFunction([&](const GraphicsBackend::Rhi::RhiRenderPassContext *ctx) {
    cmd->setPushConst(m_computePipeline, 0, 12 * sizeof(uint32_t), pc);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });
  m_computePipeline->run(cmd, 0);
  cmd->globalMemoryBarrier();
}

IFRIT_APIDECL void PostFxStockhamDFT2::renderPostFx(const GPUCmdBuffer *cmd, GPUBindId *srcSampId, GPUBindId *dstUAVImg,
                                                    uint32_t width, uint32_t height, uint32_t downscale) {
  // Round to pow of 2
  struct PushConst {
    uint32_t logW;
    uint32_t logH;
    uint32_t rtW;
    uint32_t rtH;
    uint32_t downscaleFactor;
    uint32_t rawSampId;
    uint32_t srcImgId;
    uint32_t dstImgId;
    uint32_t orientation;
    uint32_t dftMode;
  } pc;

  struct PushConstBlur {
    uint32_t alignedRtW;
    uint32_t alignedRtH;
    uint32_t srcImgId;
    uint32_t kernelImgId;
    uint32_t dstImgId;
  } pcb;

  using Ifrit::Math::FastUtil::qclz;
  using Ifrit::Math::FastUtil::qlog2;
  auto p2Width = 1 << (32 - qclz(width / downscale - 1));
  auto p2Height = 1 << (32 - qclz(height / downscale - 1));

  if (p2Width > 512 || p2Height > 512) {
    iError("Stockham DFT2: Image size too large, max 512x512");
    return;
  }

  if (m_tex1.find({p2Width, p2Height}) == m_tex1.end()) {
    auto rhi = m_app->getRhiLayer();
    auto tex1 =
        rhi->createTexture2D(p2Width, p2Height, GraphicsBackend::Rhi::RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT,
                             GraphicsBackend::Rhi::RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);
    m_tex1[{p2Width, p2Height}] = tex1;
    m_tex1Id[{p2Width, p2Height}] = rhi->registerUAVImage(tex1.get(), {0, 0, 1, 1});
  }

  auto tx1Id = m_tex1Id[{p2Width, p2Height}]->getActiveId();

  pc.logW = qlog2(p2Width);
  pc.logH = qlog2(p2Height);
  pc.rtW = width;
  pc.rtH = height;
  pc.downscaleFactor = downscale;
  pc.rawSampId = srcSampId->getActiveId();
  pc.srcImgId = tx1Id;
  pc.dstImgId = tx1Id;

  auto reqNumItersX = pc.logW;
  auto reqNumItersY = pc.logH;

  // DFT
  int wgX, wgY;

  pc.dftMode = 0;
  cmd->beginScope("Postprocess: Stockham DFT2, DFT-X");
  pc.orientation = 1;
  wgX = Ifrit::Math::ConstFunc::divRoundUp(p2Width / 2, 256);
  wgY = p2Height;
  runCommand(cmd, wgX, wgY, &pc);
  cmd->endScope();
  cmd->beginScope("Postprocess: Stockham DFT2, DFT-Y");
  pc.orientation = 0;
  wgX = Ifrit::Math::ConstFunc::divRoundUp(p2Height / 2, 256);
  wgY = p2Width;
  runCommand(cmd, wgX, wgY, &pc);
  cmd->endScope();

  // IDFT
  pc.dftMode = 1;
  cmd->beginScope("Postprocess: Stockham DFT2, IDFT-Y");
  pc.orientation = 0;
  wgX = Ifrit::Math::ConstFunc::divRoundUp(p2Height / 2, 256);
  wgY = p2Width;
  runCommand(cmd, wgX, wgY, &pc);
  cmd->endScope();
  cmd->beginScope("Postprocess: Stockham DFT2, IDFT-X");
  pc.orientation = 1;
  wgX = Ifrit::Math::ConstFunc::divRoundUp(p2Width / 2, 256);
  wgY = p2Height;
  runCommand(cmd, wgX, wgY, &pc);
  cmd->endScope();
}

} // namespace Ifrit::Core::PostprocessPassCollection