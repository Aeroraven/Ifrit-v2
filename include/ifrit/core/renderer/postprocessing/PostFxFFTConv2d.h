
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
#include "ifrit/common/util/Hash.h"
#include "ifrit/core/renderer/PostprocessPass.h"
#include <unordered_map>

namespace Ifrit::Core::PostprocessPassCollection {

struct PostFxFFTConv2dSubpasses;

struct PostFxFFTConv2dResourceCollection {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiDescHandleLegacy;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTextureRef;
  using GPUShader = Ifrit::GraphicsBackend::Rhi::RhiShader;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSamplerRef;

  GPUTexture m_tex1;
  Ref<GPUBindId> m_tex1IdSamp;
  GPUTexture m_tex2;
  GPUTexture m_texTemp;

  GPUTexture m_texGaussian;
  GPUSampler m_texGaussianSampler;
  Ref<GPUBindId> m_texGaussianSampId;

  PostFxFFTConv2dResourceCollection() = default;
  PostFxFFTConv2dResourceCollection(const PostFxFFTConv2dResourceCollection &) = default;
};

class IFRIT_APIDECL PostFxFFTConv2d : public PostprocessPass {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiDescHandleLegacy;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;

  // I know this is UGLY
  CustomHashMap<std::pair<u32, u32>, PostFxFFTConv2dResourceCollection, Ifrit::Common::Utility::PairwiseHash<u32, u32>>
      m_resMap;

  ComputePass *m_upsamplePipeline = nullptr;
  ComputePass *m_gaussianPipeline = nullptr;

public:
  PostFxFFTConv2d(IApplication *app);
  ~PostFxFFTConv2d();
  void renderPostFx(const GPUCmdBuffer *cmd, GPUBindId *srcSampId, u32 dstUAVImg, GPUBindId *kernelSampId, u32 srcWidth,
                    u32 srcHeight, u32 kernelWidth, u32 kernelHeight, u32 srcDownscale);
};

} // namespace Ifrit::Core::PostprocessPassCollection