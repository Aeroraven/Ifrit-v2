
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


struct PerFramePerViewData {
  mat4 m_worldToView;
  mat4 m_perspective;
  mat4 m_worldToClip;
  mat4 m_invPerspective;
  mat4 m_clipToWorld;
  vec4 m_cameraPosition;
  vec4 m_cameraFront;
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

float ifrit_signedDistToPlane(vec4 plane, vec4 point){
    return dot(plane.xyz,point.xyz) + plane.w;
}