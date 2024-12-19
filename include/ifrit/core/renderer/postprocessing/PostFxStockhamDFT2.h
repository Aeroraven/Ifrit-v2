
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

class IFRIT_APIDECL PostFxStockhamDFT2 : public PostprocessPass {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;

  // I know this is UGLY
  std::unordered_map<std::pair<uint32_t, uint32_t>, std::shared_ptr<GPUTexture>,
                     Ifrit::Common::Utility::PairwiseHash<uint32_t, uint32_t>>
      m_tex1;

  std::unordered_map<std::pair<uint32_t, uint32_t>, std::shared_ptr<GPUBindId>,
                     Ifrit::Common::Utility::PairwiseHash<uint32_t, uint32_t>>
      m_tex1Id;

  ComputePass *m_testBlurPipeline = nullptr;

public:
  PostFxStockhamDFT2(IApplication *app);
  void renderPostFx(const GPUCmdBuffer *cmd, GPUBindId *srcSampId,
                    GPUBindId *dstUAVImg, uint32_t width, uint32_t height,
                    uint32_t downscale);

private:
  void runCommand(const GPUCmdBuffer *cmd, uint32_t wgX, uint32_t wgY,
                  const void *pc);
};

} // namespace Ifrit::Core::PostprocessPassCollection