struct PerFramePerViewData {
  mat4 m_worldToView;
  mat4 m_perspective;
  mat4 m_invPerspective;
  vec4 m_cameraPosition;
  float m_renderWidth;
  float m_renderHeight;
  float m_cameraNear;
  float m_cameraFar;
  float m_cameraFovX;
  float m_cameraFovY;
  float m_cameraAspect;
  float m_hizLods;
};

struct PerFramePerViewDataRef{
  uint ref;
  uint pad0;
  uint pad1;
  uint pad2;
};

struct PerObjectData {
  uint transformRef;
  uint objectDataRef;
  uint instanceDataRef;
  uint transformRefLast;
  uint materialId;
};

float ifrit_recoverViewSpaceDepth(float screenZ, float nearPlane, float farPlane){
  // Near->0 (Scr), Far->1 (Scr)
  return (2.0 * nearPlane * farPlane) / ( - screenZ * (farPlane - nearPlane) + (farPlane + nearPlane));
}

