
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
#include "ifrit/common/util/Hash.h"
#include "ifrit/core/renderer/PostprocessPass.h"
#include <unordered_map>

namespace Ifrit::Core::PostprocessPassCollection {

struct PostFxFFTConv2dSubpasses;

struct PostFxFFTConv2dResourceCollection {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUShader = Ifrit::GraphicsBackend::Rhi::RhiShader;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSampler;

  std::shared_ptr<GPUTexture> m_tex1;
  std::shared_ptr<GPUBindId> m_tex1Id;
  std::shared_ptr<GPUTexture> m_tex2;
  std::shared_ptr<GPUBindId> m_tex2Id;
  std::shared_ptr<GPUTexture> m_texTemp;
  std::shared_ptr<GPUBindId> m_texTempId;

  std::shared_ptr<GPUTexture> m_texGaussian;
  std::shared_ptr<GPUSampler> m_texGaussianSampler;
  std::shared_ptr<GPUBindId> m_texGaussianId;
  std::shared_ptr<GPUBindId> m_texGaussianSampId;

  PostFxFFTConv2dResourceCollection() = default;
  PostFxFFTConv2dResourceCollection(const PostFxFFTConv2dResourceCollection &) =
      default;
};

class IFRIT_APIDECL PostFxFFTConv2d : public PostprocessPass {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;

  // I know this is UGLY
  std::unordered_map<std::pair<uint32_t, uint32_t>,
                     PostFxFFTConv2dResourceCollection,
                     Ifrit::Common::Utility::PairwiseHash<uint32_t, uint32_t>>
      m_resMap;

  ComputePass *m_upsamplePipeline = nullptr;
  ComputePass *m_gaussianPipeline = nullptr;

public:
  PostFxFFTConv2d(IApplication *app);
  ~PostFxFFTConv2d();
  void renderPostFx(const GPUCmdBuffer *cmd, GPUBindId *srcSampId,
                    GPUBindId *dstUAVImg, GPUBindId *kernelSampId,
                    uint32_t srcWidth, uint32_t srcHeight, uint32_t kernelWidth,
                    uint32_t kernelHeight, uint32_t srcDownscale);
};

} // namespace Ifrit::Core::PostprocessPassCollection