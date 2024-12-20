
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
#include "ifrit/core/renderer/postprocessing/PostFxFFTConv2d.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/math/fastutil/FastUtil.h"
#include "ifrit/core/renderer/postprocessing/PostFxStockhamDFT2.h"

namespace Ifrit::Core::PostprocessPassCollection {

IFRIT_APIDECL PostFxFFTConv2d::PostFxFFTConv2d(IApplication *app)
    : PostprocessPass(app, {"FFTConv2d.comp.glsl", 15, 1, true}) {
  using namespace Ifrit::GraphicsBackend::Rhi;
  auto rhi = app->getRhiLayer();
  m_upsamplePipeline = rhi->createComputePass();

  auto shader = createShaderFromFile("FFTConv2d.Upsample.comp.glsl", "main",
                                     RhiShaderStage::Compute);
  m_upsamplePipeline->setComputeShader(shader);
  m_upsamplePipeline->setPushConstSize(15 * sizeof(uint32_t));
  m_upsamplePipeline->setNumBindlessDescriptorSets(0);
}

IFRIT_APIDECL PostFxFFTConv2d::~PostFxFFTConv2d() {}

IFRIT_APIDECL void PostFxFFTConv2d::renderPostFx(
    const GPUCmdBuffer *cmd, GPUBindId *srcSampId, GPUBindId *dstUAVImg,
    GPUBindId *kernelSampId, uint32_t srcWidth, uint32_t srcHeight,
    uint32_t kernelWidth, uint32_t kernelHeight, uint32_t srcDownscale) {
  // todo

  auto imageX = srcWidth / srcDownscale, imageY = srcHeight / srcDownscale;
  auto imagePaddingX = kernelWidth / 2, imagePaddingY = kernelHeight / 2;

  auto imagePaddedX = imageX + imagePaddingX * 2,
       imagePaddedY = imageY + imagePaddingY * 2;

  auto kenelPaddingXLo = imagePaddedX / 2 - imageX / 2;
  auto kenelPaddingYLo = imagePaddedY / 2 - imageY / 2;
  auto kenelPaddingXHi = imagePaddedX - imageX - kenelPaddingXLo;
  auto kenelPaddingYHi = imagePaddedY - imageY - kenelPaddingYLo;

  using Ifrit::Math::FastUtil::qclz;
  using Ifrit::Math::FastUtil::qlog2;
  auto logP2Width = 32 - qclz(imagePaddedX - 1);
  auto logP2Height = 32 - qclz(imagePaddedY - 1);
  auto p2Width = 1 << logP2Width;
  auto p2Height = 1 << logP2Height;
  if (p2Width > 512 || p2Height > 512) {
    iInfo("Padded image size is: {}x{}", imagePaddedX, imagePaddedY);
    iError("FFTConv2d: Image size too large, max 512x512. Got {}x{}", p2Width,
           p2Height);
    return;
  }

  if (m_resMap.count({p2Width, p2Height}) == 0) {
    using namespace Ifrit::GraphicsBackend::Rhi;
    auto res = std::make_unique<PostFxFFTConv2dResourceCollection>();
    auto rhi = m_app->getRhiLayer();
    auto tex1 = rhi->createRenderTargetTexture(
        p2Width, p2Height, RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT);
    auto tex2 = rhi->createRenderTargetTexture(
        p2Width, p2Height, RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT);
    auto texTemp = rhi->createRenderTargetTexture(
        p2Width, p2Height, RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT);
    auto tex1Id = rhi->registerUAVImage(tex1.get(), {0, 0, 1, 1});
    auto tex2Id = rhi->registerUAVImage(tex2.get(), {0, 0, 1, 1});
    auto texTempId = rhi->registerUAVImage(texTemp.get(), {0, 0, 1, 1});
    res->m_tex1 = tex1;
    res->m_tex1Id = tex1Id;
    res->m_tex2 = tex2;
    res->m_tex2Id = tex2Id;
    res->m_texTemp = texTemp;
    res->m_texTempId = texTempId;
    m_resMap[{p2Width, p2Height}] = *res;
  }

  struct PushConst {
    uint32_t srcDownScale;
    uint32_t kernDownScale;
    uint32_t srcRtW;
    uint32_t srcRtH;
    uint32_t kernRtW;
    uint32_t kernRtH;
    uint32_t srcImage;
    uint32_t srcIntermImage;
    uint32_t kernImage;
    uint32_t kernIntermImage;
    uint32_t dstImage;
    uint32_t tempImage;
    uint32_t fftTexSizeWLog;
    uint32_t fftTexSizeHLog;
    uint32_t fftStep;
  } pc;

  pc.srcDownScale = srcDownscale;
  pc.kernDownScale = 1;
  pc.srcRtW = srcWidth;
  pc.srcRtH = srcHeight;
  pc.kernRtW = kernelWidth;
  pc.kernRtH = kernelHeight;
  pc.srcImage = srcSampId->getActiveId();
  pc.srcIntermImage = m_resMap[{p2Width, p2Height}].m_tex1Id->getActiveId();
  pc.kernImage = 0; // kernelSampId->getActiveId();
  pc.kernIntermImage = m_resMap[{p2Width, p2Height}].m_tex2Id->getActiveId();
  pc.dstImage = dstUAVImg->getActiveId();
  pc.tempImage = m_resMap[{p2Width, p2Height}].m_texTempId->getActiveId();
  pc.fftTexSizeWLog = logP2Width;
  pc.fftTexSizeHLog = logP2Height;

  if (logP2Height != logP2Width) {
    iError("FFTConv2d: Image size must be square");
    return;
  }
  auto wgX = Ifrit::Math::ConstFunc::divRoundUp(p2Width / 2, 256);
  auto wgY = p2Height;
  cmd->beginScope("Postprocess: FFTConv2D");
  const char *scopeNames[7] = {
      "Postprocess: FFTConv2D, SrcFFT-X",  "Postprocess: FFTConv2D, SrcFFT-Y",
      "Postprocess: FFTConv2D, KernFFT-X", "Postprocess: FFTConv2D, KernFFT-Y",
      "Postprocess: FFTConv2D, Multiply",  "Postprocess: FFTConv2D, IFFT-X",
      "Postprocess: FFTConv2D, IFFT-Y",
  };
  for (uint32_t i = 0u; i < 7u; i++) {
    cmd->beginScope(scopeNames[i]);
    m_computePipeline->setRecordFunction(
        [&](const GraphicsBackend::Rhi::RhiRenderPassContext *ctx) {
          pc.fftStep = i;
          cmd->setPushConst(m_computePipeline, 0, 15 * sizeof(uint32_t), &pc);
          ctx->m_cmd->dispatch(wgX, wgY, 1);
        });
    m_computePipeline->run(cmd, 0);
    cmd->globalMemoryBarrier();
    cmd->endScope();
  }

  cmd->beginScope("Postprocess: FFTConv2D, Upsample");
  auto dwgX = Ifrit::Math::ConstFunc::divRoundUp(srcWidth, 8);
  auto dwgY = Ifrit::Math::ConstFunc::divRoundUp(srcHeight, 8);
  m_upsamplePipeline->setRecordFunction(
      [&](const GraphicsBackend::Rhi::RhiRenderPassContext *ctx) {
        pc.tempImage = m_resMap[{p2Width, p2Height}].m_texTempId->getActiveId();
        cmd->setPushConst(m_upsamplePipeline, 0, 15 * sizeof(uint32_t), &pc);
        ctx->m_cmd->dispatch(dwgX, dwgY, 1);
      });
  m_upsamplePipeline->run(cmd, 0);
  cmd->globalMemoryBarrier();
  cmd->endScope();
  cmd->endScope();
}
} // namespace Ifrit::Core::PostprocessPassCollection