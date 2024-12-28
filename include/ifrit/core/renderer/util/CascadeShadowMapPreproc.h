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
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/core/renderer/RendererBase.h"
#include "ifrit/core/scene/FrameCollector.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <algorithm>
#include <bit>

#include "ifrit/core/base/Light.h"

namespace Ifrit::Core::RenderingUtil::CascadeShadowMapping {

constexpr uint32_t kDefaultCSMCount = 4;
constexpr std::array<float, 4> kDefaultCSMSplits = {0.067f, 0.133f, 0.267f,
                                                    0.533f};
constexpr std::array<float, 4> kDefaultCSMBorders = {0.08f, 0.05f, 0.0f, 0.0f};

struct CSMSingleSplitResult {
  float4x4 m_proj;
  float4x4 m_view;
  ifloat4 m_lightCamPos;
  float m_orthoSize;
  float m_near;
  float m_far;
};

struct CSMResult {
  std::vector<CSMSingleSplitResult> m_splits;
  std::array<float, 4> m_splitStart;
  std::array<float, 4> m_splitEnd;
};

IFRIT_APIDECL CSMResult calculateCSMSplits(
    const Ifrit::Core::PerFrameData::PerViewData &perView,
    uint32_t shadowResolution, ifloat3 lightFront, uint32_t splitCount,
    float maxDistance, const std::vector<float> &splits,
    const std::vector<float> &borders);

IFRIT_APIDECL std::vector<PerFrameData::PerViewData>
fillCSMViews(const Ifrit::Core::PerFrameData::PerViewData &perView,
             Light &light, uint32_t shadowResolution, Transform &lightTransform,
             uint32_t splitCount, float maxDistance,
             const std::vector<float> &splits,
             const std::vector<float> &borders,
             std::array<float, 4> &splitStart, std::array<float, 4> &splitEnd);

} // namespace Ifrit::Core::RenderingUtil::CascadeShadowMapping