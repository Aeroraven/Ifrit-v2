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

IFRIT_APIDECL CSMResult calculateCSMSplits(const PerFrameData::PerViewData &perView, u32 shadowResolution,
                                           Vector3f lightFront, u32 splitCount, float maxDistance,
                                           const Vec<float> &splits, const Vec<float> &borders) {

  using namespace Ifrit::Math;

  Logging::assertion(splitCount <= 4, "Split count should be less than 4");

  // Prepare splits
  Vec<float> splitStartMeter;
  Vec<float> splitEndMeter;

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
  Vec<Vector4f> boundSpheres;
  auto camAspect = perView.m_viewData.m_cameraAspect;
  auto camFovY = perView.m_viewData.m_cameraFovY;
  auto camPos = perView.m_viewData.m_cameraPosition;

  // Bound AABBs
  Vec<CSMSingleSplitResult> results;
  for (auto i = 0u; i < splitCount; i++) {
    auto vNear = std::max(1e-4f, splitStartMeter[i]);
    auto vFar = splitEndMeter[i];
    auto vApex = Vector3f{camPos.x, camPos.y, camPos.z};
    auto rZFar = 0.0f, rOrthoSize = 0.0f;
    auto rCullOrthoX = 0.0f, rCullOrthoY = 0.0f;
    Vector3f rCenter;
    auto worldToView = Math::transpose(perView.m_viewData.m_worldToView);
    auto viewToWorld = inverse4((worldToView));
    getFrustumBoundingBoxWithRay(camFovY, camAspect, vNear, vFar, viewToWorld, vApex, lightFront, 1e2f, rZFar,
                                 rOrthoSize, rCenter, rCullOrthoX, rCullOrthoY);
    auto lightCamUp = Vector3f{0.0f, 1.0f, 0.0f};
    auto proj = orthographicNegateY(rOrthoSize, 1.0, 0.1f, rZFar);
    auto view = lookAt(rCenter, rCenter + lightFront, lightCamUp);

    // To alleviate shadow flickering, we need to snap the light camera to texel
    auto viewOriginal = lookAt(Vector3f{0.0f, 0.0f, 0.0f}, lightFront, lightCamUp);
    auto viewOriginalInv = inverse4(viewOriginal);
    auto camPosOriginal = matmul(viewOriginal, Vector4f{rCenter.x, rCenter.y, rCenter.z, 1.0f});
    float texelSize = rOrthoSize / shadowResolution;
    auto camPosSnapped = Vector3f{camPosOriginal.x - std::fmodf(camPosOriginal.x, texelSize),
                                  camPosOriginal.y - std::fmodf(camPosOriginal.y, texelSize),
                                  camPosOriginal.z - std::fmodf(camPosOriginal.z, texelSize)};
    auto camPosSnappedWorld =
        matmul(viewOriginalInv, Vector4f{camPosSnapped.x, camPosSnapped.y, camPosSnapped.z, 1.0f});
    auto newCenter = Vector3f{camPosSnappedWorld.x, camPosSnappedWorld.y, camPosSnappedWorld.z};
    view = lookAt(newCenter, newCenter + lightFront, lightCamUp);

    CSMSingleSplitResult result;
    result.m_proj = proj;
    result.m_view = view;
    result.m_lightCamPos = {rCenter.x, rCenter.y, rCenter.z, 1.0f};
    result.m_near = 1e-3;
    result.m_far = rZFar;
    result.m_orthoSize = rOrthoSize;
    result.m_clipOrthoSizeX = rCullOrthoX;
    result.m_clipOrthoSizeY = rCullOrthoY;
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

IFRIT_APIDECL Vec<PerFrameData::PerViewData> fillCSMViews(const PerFrameData::PerViewData &perView, Light &light,
                                                          u32 shadowResolution, Transform &lightTransform,
                                                          u32 splitCount, float maxDistance, const Vec<float> &splits,
                                                          const Vec<float> &borders, std::array<float, 4> &splitStart,
                                                          std::array<float, 4> &splitEnd) {
  auto lightTransformMat = lightTransform.getModelToWorldMatrix();
  auto lightDirRaw = Vector4f{0.0f, 0.0f, 1.0f, 0.0f};
  auto lightDir = Math::matmul(lightTransformMat, lightDirRaw);
  auto lightDir3 = Vector3f{lightDir.x, lightDir.y, lightDir.z};
  auto csmResult = calculateCSMSplits(perView, shadowResolution, lightDir3, splitCount, maxDistance, splits, borders);

  Vec<PerFrameData::PerViewData> views;
  for (auto i = 0u; i < splitCount; i++) {
    auto &split = csmResult.m_splits[i];
    auto view = perView;
    view.m_viewType = PerFrameData::ViewType::Shadow;
    view.m_viewData.m_worldToView = Math::transpose(split.m_view);
    view.m_viewData.m_perspective = Math::transpose(split.m_proj);
    view.m_viewData.m_worldToClip = Math::transpose(
        Math::matmul(Math::transpose(view.m_viewData.m_perspective), Math::transpose(view.m_viewData.m_worldToView)));
    view.m_viewData.m_cameraAspect = 1.0f;
    view.m_viewData.m_inversePerspective =
        Math::transpose(Ifrit::Math::inverse4(Math::transpose(view.m_viewData.m_perspective)));
    view.m_viewData.m_clipToWorld = Math::transpose(Math::inverse4(Math::transpose(view.m_viewData.m_worldToClip)));
    view.m_viewData.m_cameraPosition = split.m_lightCamPos;
    view.m_viewData.m_cameraFront = {lightDir.x, lightDir.y, lightDir.z, 0.0f};
    view.m_viewData.m_cameraNear = split.m_near;
    view.m_viewData.m_cameraFar = split.m_far;
    view.m_viewData.m_cameraFovX = 0.0f;
    view.m_viewData.m_cameraFovY = 0.0f;
    view.m_viewData.m_cameraOrthoSize = split.m_orthoSize;
    view.m_viewData.m_cullCamOrthoSizeX = split.m_clipOrthoSizeX;
    view.m_viewData.m_cullCamOrthoSizeY = split.m_clipOrthoSizeY;
    view.m_viewData.m_viewCameraType = 1.0f;
    view.m_viewData.m_viewToWorld = Math::transpose(Math::inverse4(Math::transpose(view.m_viewData.m_worldToView)));

    auto shadowMapSize = light.getShadowMapResolution();
    view.m_viewData.m_renderHeightf = static_cast<float>(shadowMapSize);
    view.m_viewData.m_renderWidthf = static_cast<float>(shadowMapSize);
    view.m_viewData.m_hizLods = std::floor(std::log2(1.0f * shadowMapSize)) + 1.0f;

    views.push_back(view);
  }
  splitStart = csmResult.m_splitStart;
  splitEnd = csmResult.m_splitEnd;
  return views;
}

} // namespace Ifrit::Core::RenderingUtil::CascadeShadowMapping