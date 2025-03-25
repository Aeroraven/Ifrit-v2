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

IF_CONSTEXPR u32 kDefaultCSMCount = 4;
IF_CONSTEXPR Array<f32, 4> kDefaultCSMSplits = {0.067f, 0.133f, 0.267f, 0.533f};
IF_CONSTEXPR Array<f32, 4> kDefaultCSMBorders = {0.08f, 0.05f, 0.0f, 0.0f};

struct CSMSingleSplitResult {
  Matrix4x4f m_proj;
  Matrix4x4f m_view;
  Vector4f m_lightCamPos;
  f32 m_orthoSize;
  f32 m_near;
  f32 m_far;

  f32 m_clipOrthoSizeX;
  f32 m_clipOrthoSizeY;
};

struct CSMResult {
  Vec<CSMSingleSplitResult> m_splits;
  Array<f32, 4> m_splitStart;
  Array<f32, 4> m_splitEnd;
};

IFRIT_APIDECL CSMResult calculateCSMSplits(const Ifrit::Core::PerFrameData::PerViewData &perView, u32 shadowResolution,
                                           Vector3f lightFront, u32 splitCount, f32 maxDistance, const Vec<f32> &splits,
                                           const Vec<f32> &borders);

IFRIT_APIDECL Vec<PerFrameData::PerViewData> fillCSMViews(const Ifrit::Core::PerFrameData::PerViewData &perView,
                                                          Light &light, u32 shadowResolution, Transform &lightTransform,
                                                          u32 splitCount, f32 maxDistance, const Vec<f32> &splits,
                                                          const Vec<f32> &borders, Array<f32, 4> &splitStart,
                                                          Array<f32, 4> &splitEnd);

} // namespace Ifrit::Core::RenderingUtil::CascadeShadowMapping