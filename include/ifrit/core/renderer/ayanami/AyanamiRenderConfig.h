
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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

namespace Ifrit::Core::Ayanami {
struct AyanamiRenderConfig {
  u32 m_globalDFClipmapLevels = 4;
  u32 m_globalDFClipmapResolution = 256; // 16MB per clipmap level
  f32 m_globalDFBaseExtent = 20.0f;      // 2500.0 in the original code
};
} // namespace Ifrit::Core::Ayanami