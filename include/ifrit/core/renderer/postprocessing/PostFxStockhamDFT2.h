
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

class IFRIT_APIDECL PostFxStockhamDFT2 : public PostprocessPass {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTextureRef;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;

  // I know this is UGLY
  std::unordered_map<std::pair<u32, u32>, GPUTexture, Ifrit::Common::Utility::PairwiseHash<u32, u32>> m_tex1;

  std::unordered_map<std::pair<u32, u32>, std::shared_ptr<GPUBindId>, Ifrit::Common::Utility::PairwiseHash<u32, u32>>
      m_tex1Id;

  ComputePass *m_testBlurPipeline = nullptr;

public:
  PostFxStockhamDFT2(IApplication *app);
  void renderPostFx(const GPUCmdBuffer *cmd, GPUBindId *srcSampId, GPUBindId *dstUAVImg, u32 width, u32 height,
                    u32 downscale);

private:
  void runCommand(const GPUCmdBuffer *cmd, u32 wgX, u32 wgY, const void *pc);
};

} // namespace Ifrit::Core::PostprocessPassCollection