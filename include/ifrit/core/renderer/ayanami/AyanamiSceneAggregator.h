
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

#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/scene/FrameCollector.h"
#include "ifrit/core/scene/SceneManager.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core::Ayanami {

struct AyanamiSceneResources;

class IFRIT_APIDECL AyanamiSceneAggregator : public Common::Utility::NonCopyable {
private:
  GraphicsBackend::Rhi::RhiBackend *m_rhi;
  AyanamiSceneResources *m_sceneResources = nullptr;

private:
  void init();
  void destroy();

public:
  AyanamiSceneAggregator(GraphicsBackend::Rhi::RhiBackend *rhi) : m_rhi(rhi) { init(); }
  ~AyanamiSceneAggregator() { destroy(); }

  void collectScene(Scene *scene);
  u32 getGatheredBufferId();
};

} // namespace Ifrit::Core::Ayanami
