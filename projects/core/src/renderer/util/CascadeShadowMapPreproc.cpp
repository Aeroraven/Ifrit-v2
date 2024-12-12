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

#include "ifrit/core/renderer/util/CascadeShadowMapPreproc.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/GeometryFunctions.h"

namespace Ifrit::Core::RenderingUtil::CascadeShadowMapping {

IFRIT_APIDECL CSMResult calculateCSMSplits(
    const PerFrameData::PerViewData &perView, ifloat3 lightFront,
    uint32_t splitCount, float maxDistance, const std::vector<float> &splits,
    const std::vector<float> &borders) {

  using namespace Ifrit::Math;

  Logging::assertion(splitCount <= 4, "Split count should be less than 4");

  // Prepare splits
  std::vector<float> splitStartMeter;
  std::vector<float> splitEndMeter;

  float accumSplit = 0.0f;
  for (auto i = 0u; i < splitCount; i++) {
    if (i == 0u) {
      splitStartMeter.push_back(0.0f);
    } else {
      auto borderPercent = borders[i - 1] * splits[i - 1];
      auto borderMeter = borderPercent * maxDistance;
      splitStartMeter.push_back(splitEndMeter[i - 1] - borderMeter);
    }
    accumSplit += splits[i];
    splitEndMeter.push_back(accumSplit * maxDistance);
  }

  // Calculate frustum bounding spheres
  std::vector<ifloat4> boundSpheres;
  auto camAspect = perView.m_viewData.m_cameraFovY;
  auto camFovY = perView.m_viewData.m_cameraAspect;
  auto camPos = perView.m_viewData.m_cameraPosition;

  for (auto i = 0u; i < splitCount; i++) {
    auto vNear = splitStartMeter[i];
    auto vFar = splitEndMeter[i];
    auto vApex = ifloat3{camPos.x, camPos.y, camPos.z};
    auto sphere =
        getFrustumBoundingSphere(camFovY, camAspect, vNear, vFar, vApex);
    boundSpheres.push_back(sphere);
  }

  // Make result
  std::vector<CSMSingleSplitResult> results;
  for (auto i = 0u; i < splitCount; i++) {
    auto &sphere = boundSpheres[i];
    sphere.w += 1e-3f;
    auto lightCamPosShiftXN =
        ifloat3{-lightFront.x * sphere.w, -lightFront.y * sphere.w,
                -lightFront.z * sphere.w};
    auto lightCamPosShiftXF =
        ifloat3{lightFront.x * sphere.w, lightFront.y * sphere.w,
                lightFront.z * sphere.w};
    auto lightCamPos = ifloat3{lightCamPosShiftXN.x + sphere.x,
                               lightCamPosShiftXN.y + sphere.y,
                               lightCamPosShiftXN.z + sphere.z};
    auto lightCamLookAtCenter = ifloat3{sphere.x, sphere.y, sphere.z};
    auto lightCamUp = ifloat3{0.0f, 1.0f, 0.0f};
    auto proj = orthographicNegateY(sphere.w * 2, 1.0, 1e-3f, 2 * sphere.w);
    auto view = lookAt(lightCamPos, lightCamLookAtCenter, lightCamUp);

    CSMSingleSplitResult result;
    result.m_proj = proj;
    result.m_view = view;
    result.m_lightCamPos = {lightCamPos.x, lightCamPos.y, lightCamPos.z, 1.0f};
    result.m_near = 1e-3f;
    result.m_far = 2 * sphere.w;
    result.m_orthoSize = sphere.w * 2;
    results.push_back(result);
  }

  CSMResult resultf;
  resultf.m_splits = results;
  for (auto i = 0u; i < splitCount; i++) {
    resultf.m_splitStart[i] = splitStartMeter[i];
    resultf.m_splitEnd[i] = splitEndMeter[i];
  }

  return resultf;
}

IFRIT_APIDECL std::vector<PerFrameData::PerViewData>
fillCSMViews(const PerFrameData::PerViewData &perView, Light &light,
             Transform &lightTransform, uint32_t splitCount, float maxDistance,
             const std::vector<float> &splits,
             const std::vector<float> &borders,
             std::array<float, 4> &splitStart, std::array<float, 4> &splitEnd) {
  auto lightTransformMat = lightTransform.getModelToWorldMatrix();
  auto lightDirRaw = ifloat4{0.0f, 0.0f, 1.0f, 0.0f};
  auto lightDir = Math::matmul(lightTransformMat, lightDirRaw);

  auto lightDir3 = ifloat3{lightDir.x, lightDir.y, lightDir.z};
  auto csmResult = calculateCSMSplits(perView, lightDir3, splitCount,
                                      maxDistance, splits, borders);

  std::vector<PerFrameData::PerViewData> views;
  for (auto i = 0u; i < splitCount; i++) {
    auto &split = csmResult.m_splits[i];
    auto view = perView;
    view.m_viewType = PerFrameData::ViewType::Shadow;
    view.m_viewData.m_worldToView = Math::transpose(split.m_view);
    view.m_viewData.m_perspective = Math::transpose(split.m_proj);
    view.m_viewData.m_worldToClip = Math::transpose(
        Math::matmul(Math::transpose(view.m_viewData.m_perspective),
                     Math::transpose(view.m_viewData.m_worldToView)));
    view.m_viewData.m_cameraAspect = 1.0f;
    view.m_viewData.m_inversePerspective =
        Ifrit::Math::inverse4(view.m_viewData.m_perspective);
    view.m_viewData.m_clipToWorld =
        Math::inverse4(view.m_viewData.m_worldToClip);
    view.m_viewData.m_cameraPosition = split.m_lightCamPos;
    view.m_viewData.m_cameraFront = {lightDir.x, lightDir.y, lightDir.z, 0.0f};
    view.m_viewData.m_cameraNear = split.m_near;
    view.m_viewData.m_cameraFar = split.m_far;
    view.m_viewData.m_cameraFovX = 0.0f;
    view.m_viewData.m_cameraFovY = 0.0f;
    view.m_viewData.m_cameraOrthoSize = split.m_orthoSize;
    view.m_viewData.m_viewCameraType = 1.0f;
    view.m_viewData.m_viewToWorld =
        Math::inverse4(view.m_viewData.m_worldToView);

    auto shadowMapSize = light.getShadowMapResolution();
    view.m_viewData.m_renderHeightf = static_cast<float>(shadowMapSize);
    view.m_viewData.m_renderWidthf = static_cast<float>(shadowMapSize);
    view.m_viewData.m_hizLods =
        std::floor(std::log2(1.0f * shadowMapSize)) + 1.0f;

    views.push_back(view);
  }
  splitStart = csmResult.m_splitStart;
  splitEnd = csmResult.m_splitEnd;
  return views;
}

} // namespace Ifrit::Core::RenderingUtil::CascadeShadowMapping