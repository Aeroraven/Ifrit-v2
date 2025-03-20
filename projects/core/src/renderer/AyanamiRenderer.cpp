
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

#include "ifrit/core/renderer/AyanamiRenderer.h"

namespace Ifrit::Core {

struct AyanamiRendererResources {};

IFRIT_APIDECL void AyanamiRenderer::initRenderer() { m_resources = new AyanamiRendererResources(); }
IFRIT_APIDECL AyanamiRenderer::~AyanamiRenderer() {
  if (m_resources)
    delete m_resources;
}

IFRIT_APIDECL std::unique_ptr<AyanamiRenderer::GPUCommandSubmission>
AyanamiRenderer::render(Scene *scene, Camera *camera, RenderTargets *renderTargets, const RendererConfig &config,
                        const std::vector<GPUCommandSubmission *> &cmdToWait) {

  // Generate GBuffer first
  auto gbufferTask = m_gbufferRenderer->render(scene, camera, renderTargets, config, cmdToWait);
  auto perframeData = m_gbufferRenderer->getPerframeData(scene);
  return nullptr;
}

} // namespace Ifrit::Core