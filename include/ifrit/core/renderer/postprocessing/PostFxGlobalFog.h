
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
#include "ifrit/core/renderer/PostprocessPass.h"

namespace Ifrit::Core::PostprocessPassCollection {

class IFRIT_APIDECL PostFxGlobalFog : public PostprocessPass {
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiDescHandleLegacy;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;

public:
  PostFxGlobalFog(IApplication *app);
  void renderPostFx(const GPUCmdBuffer *cmd, RenderTargets *renderTargets, GPUBindId *inputTexCombSampler,
                    GPUBindId *inputDepthTexCombSampler, GPUBindId *inputViewUniform);
};

} // namespace Ifrit::Core::PostprocessPassCollection