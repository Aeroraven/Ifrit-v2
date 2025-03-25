
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
#include <array>
namespace Ifrit::Core {

struct RendererConsts {
  static IF_CONSTEXPR Array<f32, 8> cHalton2 = {0.5f, 0.25f, 0.75f, 0.125f, 0.625f, 0.375f, 0.875f, 0.0625f};
  static IF_CONSTEXPR Array<f32, 8> cHalton3 = {0.3333333333333333f, 0.6666666666666666f, 0.1111111111111111f,
                                                0.4444444444444444f, 0.7777777777777778f, 0.2222222222222222f,
                                                0.5555555555555556f, 0.8888888888888888f};
};

} // namespace Ifrit::Core