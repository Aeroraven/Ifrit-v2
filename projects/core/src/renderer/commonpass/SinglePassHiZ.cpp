#include "ifrit/common/base/IfritBase.h"

#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/core/renderer/commonpass/SinglePassHiZ.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"

namespace Ifrit::Core {
using namespace RenderingUtil;
using namespace GraphicsBackend::Rhi;
using namespace Math;

IF_CONSTEXPR auto kbImUsage_UAV_SRV = RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_CopyDest =
    RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_TRANSFER_DST_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_RT_CopySrc = RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT |
                                                 RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                 RHI_IMAGE_USAGE_TRANSFER_SRC_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_RT =
    RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
IF_CONSTEXPR auto kbImUsage_SRV_DEPTH = RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
IF_CONSTEXPR auto kbImUsage_UAV = RHI_IMAGE_USAGE_STORAGE_BIT;

IF_CONSTEXPR auto kbBufUsage_SSBO_CopyDest = RhiBufferUsage_SSBO | RhiBufferUsage_CopyDst;

IFRIT_APIDECL SinglePassHiZPass::SinglePassHiZPass(IApplication *app) {
  auto rhi = app->getRhiLayer();
  m_app = app;
  m_singlePassHiZPass = createComputePass(rhi, "CommonPass/SinglePassHzb.comp.glsl", 1, 6);
}

IFRIT_APIDECL bool SinglePassHiZPass::checkResourceToRebuild(PerFrameData::SinglePassHiZData &data, u32 rtWidth,
                                                             u32 rtHeight) {
  bool cond = (data.m_hizTexture == nullptr);

  // Make width and heights power of 2
  auto rWidth = 1 << (int)std::ceil(std::log2(rtWidth));
  auto rHeight = 1 << (int)std::ceil(std::log2(rtHeight));
  if (!cond && (data.m_hizTexture->getWidth() != rWidth || data.m_hizTexture->getHeight() != rHeight)) {
    cond = true;
  }
  if (!cond)
    return false;
  return true;
}

IFRIT_APIDECL void SinglePassHiZPass::prepareHiZResources(PerFrameData::SinglePassHiZData &data,
                                                          GPUTexture *depthTexture, GPUSampler *sampler, u32 rtWidth,
                                                          u32 rtHeight) {
  auto rhi = m_app->getRhiLayer();
  auto maxMip = int(std::floor(std::log2(std::max(rtWidth, rtHeight))) + 1);
  data.m_hizTexture = rhi->createMipMapTexture("SHiZ_Tex", rtWidth, rtHeight, maxMip,
                                               RhiImageFormat::RhiImgFmt_R32_SFLOAT, kbImUsage_UAV, true);

  data.m_hizRefs.resize(0);
  data.m_hizRefs.push_back(0);
  for (int i = 0; i < maxMip; i++) {
    auto bindlessId = rhi->registerUAVImage2(data.m_hizTexture.get(), {static_cast<u32>(i), 0, 1, 1});
    data.m_hizRefs.push_back(bindlessId->getActiveId());
  }
  data.m_hizRefBuffer =
      rhi->createBufferDevice("SHiZ_Ref", u32Size * data.m_hizRefs.size(), kbBufUsage_SSBO_CopyDest, true);
  data.m_hizAtomics =
      rhi->createBufferDevice("SHiZ_Atmoics", u32Size * data.m_hizRefs.size(), kbBufUsage_SSBO_CopyDest, true);
  auto staged = rhi->createStagedSingleBuffer(data.m_hizRefBuffer.get());
  auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
  tq->runSyncCommand([&](const GPUCmdBuffer *cmd) {
    staged->cmdCopyToDevice(cmd, data.m_hizRefs.data(), u32Size * data.m_hizRefs.size(), 0);
  });

  data.m_hizDesc = rhi->createBindlessDescriptorRef();
  data.m_hizDesc->addStorageBuffer(data.m_hizRefBuffer.get(), 1);
  data.m_hizDesc->addCombinedImageSampler(depthTexture, sampler, 0);
  data.m_hizDesc->addStorageBuffer(data.m_hizAtomics.get(), 2);

  data.m_hizWidth = rtWidth;
  data.m_hizHeight = rtHeight;
  data.m_hizIters = maxMip;
}

IFRIT_APIDECL void SinglePassHiZPass::runHiZPass(const PerFrameData::SinglePassHiZData &data, const GPUCmdBuffer *cmd,
                                                 u32 rtWidth, u32 rtHeight, bool minMode) {
  struct SpHizPushConst {
    u32 m_hizWidth;
    u32 m_hizHeight;
    u32 m_rtWidth;
    u32 m_rtHeight;
    u32 m_hizIters;
    u32 m_minMode;
  } pc;
  pc.m_hizHeight = data.m_hizHeight;
  pc.m_hizWidth = data.m_hizWidth;
  pc.m_rtWidth = rtWidth;
  pc.m_rtHeight = rtHeight;
  pc.m_hizIters = data.m_hizIters;
  pc.m_minMode = minMode ? 1 : 0;

  IF_CONSTEXPR static u32 cSPHiZTileSize = 64;

  m_singlePassHiZPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceCompute(m_singlePassHiZPass, 1, data.m_hizDesc);
    ctx->m_cmd->setPushConst(m_singlePassHiZPass, 0, u32Size * 6, &pc);
    auto tgX = divRoundUp(data.m_hizWidth, cSPHiZTileSize);
    auto tgY = divRoundUp(data.m_hizHeight, cSPHiZTileSize);
    ctx->m_cmd->dispatch(tgX, tgY, 1);
  });
  cmd->beginScope("Single Pass Hierarchical Z-Buffer");
  m_singlePassHiZPass->run(cmd, 0);
  cmd->endScope();
}
} // namespace Ifrit::Core