
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
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/core/renderer/RendererUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <algorithm>
#include <bit>

namespace Ifrit::Core::RenderingUtil {
IFRIT_APIDECL GraphicsBackend::Rhi::RhiTextureRef loadBlueNoise(GraphicsBackend::Rhi::RhiBackend *rhi);
} // namespace Ifrit::Core::RenderingUtil