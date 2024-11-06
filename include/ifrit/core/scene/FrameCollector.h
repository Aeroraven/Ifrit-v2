#pragma once
#include "ifrit/common/math/VectorDefs.h"
namespace Ifrit::Core {

struct PerFramePerViewData {
  float4x4 m_worldToView;
  float4x4 m_perspective;
  ifloat4 m_cameraPosition;
  float m_renderWidth;
  float m_renderHeight;
  float m_cameraNear;
  float m_cameraFar;
  float m_cameraFovX;
  float m_cameraFovY;
};

struct PerObjectData {
  float4x4 m_modelToWorld;
};

} // namespace Ifrit::Core