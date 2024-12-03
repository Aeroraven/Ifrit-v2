
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


#include "ifrit/core/scene/SceneManager.h"
#include "ifrit/common/util/TypingUtil.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::Core {
IFRIT_APIDECL void
SceneManager::collectPerframeData(PerFrameData &perframeData, Scene *scene,
                                  Camera *camera,
                                  GraphicsShaderPassType passType) {
  throw std::runtime_error("Deprecated");
}
} // namespace Ifrit::Core