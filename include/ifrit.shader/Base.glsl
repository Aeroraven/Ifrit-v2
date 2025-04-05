
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
  mat4 m_viewToWorld;
  vec4 m_cameraPosition;
  vec4 m_cameraFront;
  float m_renderWidth;
  float m_renderHeight;
  float m_cameraNear;
  float m_cameraFar;
  float m_cameraFovX;
  float m_cameraFovY;
  float m_cameraAspect;
  float m_cameraOrthoSize;
  float m_hizLods;
  float m_viewCameraType;

  float m_cullCamOrthoSizeX;
  float m_cullCamOrthoSizeY;
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

float ifrit_viewZToClipZ(float viewZ, float zNear, float zFar){
  float dz = zFar/(zFar-zNear)*viewZ - zFar*zNear/(zFar-zNear);
  return dz/viewZ;
}

float ifrit_clipZToViewZ(float clipZ, float zNear, float zFar){
  // w =(f/(f-n)*z - f*n/(f-n))/z
  // w = f/(f-n) - f*n/(f-n)/z
  // zw = fz/(f-n) - f*n/(f-n)
  // fn/(f-n) = z(f/(f-n) - w)
  // z = fn/(f-n)/(f/(f-n) - w)
  

  float fn_mul = zFar*zNear;
  float fn_sub = zFar-zNear;  
  return fn_mul/(zFar-clipZ*fn_sub);
}

float ifrit_signedDistToPlane(vec4 plane, vec4 point){
    return dot(plane.xyz,point.xyz) + plane.w;
}


float ifrit_perspectiveLerp(float v0, float v1, float z0, float z1, float t){
  float lp = (1.0/z0 + (1.0/z1 - 1.0/z0) * t);
  return (v0/z0 + (v1/z1 - v0/z0) * t) / lp;
}

vec2 ifrit_perspectiveLerp2D(vec2 v0, vec2 v1, float z0, float z1, float t){
  float lp = (1.0/z0 + (1.0/z1 - 1.0/z0) * t);
  return (v0/z0 + (v1/z1 - v0/z0) * t) / lp;
}

vec3 ifrit_perspectiveLerp3D(vec3 v0, vec3 v1, float z0, float z1, float t){
  float lp = (1.0/z0 + (1.0/z1 - 1.0/z0) * t);
  return (v0/z0 + (v1/z1 - v0/z0) * t) / lp;
}

vec4 ifrit_perspectiveLerp4D(vec4 v0, vec4 v1, float z0, float z1, float t){
  float lp = (1.0/z0 + (1.0/z1 - 1.0/z0) * t);
  return (v0/z0 + (v1/z1 - v0/z0) * t) / lp;
}

bool ifrit_RayboxIntersection(vec3 o,vec3 d,vec3 lb,vec3 rt, out float t){
    // from: https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    vec3 dirfrac;
    dirfrac.x = 1.0f / d.x;
    dirfrac.y = 1.0f / d.y;
    dirfrac.z = 1.0f / d.z;
    float t1 = (lb.x - o.x)*dirfrac.x;
    float t2 = (rt.x - o.x)*dirfrac.x;
    float t3 = (lb.y - o.y)*dirfrac.y;
    float t4 = (rt.y - o.y)*dirfrac.y;
    float t5 = (lb.z - o.z)*dirfrac.z;
    float t6 = (rt.z - o.z)*dirfrac.z;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
    if (tmax < 0){
        t = tmax;
        return false;
    }
    if (tmin > tmax){
        t = tmax;
        return false;
    }
    t = tmin;
    return true;
}


const float kPI = 3.14159265359;