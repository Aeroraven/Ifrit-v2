struct PerFramePerViewData {
  mat4 m_worldToView;
  mat4 m_perspective;
  vec4 m_cameraPosition;
  float m_renderWidth;
  float m_renderHeight;
  float m_cameraNear;
  float m_cameraFar;
  float m_cameraFovX;
  float m_cameraFovY;
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
  uint pad0;
  uint pad1;
};

