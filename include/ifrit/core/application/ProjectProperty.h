
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

namespace Ifrit::Core {
enum class ApplicationRhiType { Vulkan, DX12, OpenGL, Software };
enum class ApplicationDisplayProvider { GLFW };

struct ProjectProperty {
  String m_name;
  String m_version;

  ApplicationRhiType m_rhiType = ApplicationRhiType::Vulkan;
  ApplicationDisplayProvider m_displayProvider = ApplicationDisplayProvider::GLFW;

  u32 m_width = 1980;
  u32 m_height = 1080;

  String m_assetPath;
  String m_scenePath;
  String m_cachePath;

  u32 m_rhiGraphicsQueueCount = 1;
  u32 m_rhiTransferQueueCount = 1;
  u32 m_rhiComputeQueueCount = 1;
  u32 m_rhiNumBackBuffers = 2;
  u32 m_rhiDebugMode = 0;

  u32 m_fixedUpdateRate = 20000;           // 0.02s
  u32 m_fixedUpdateCompensationLimit = 10; // allow max 10 frames
};

} // namespace Ifrit::Core