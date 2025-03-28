
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

#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Ayanami/Ayanami.Shared.glsl"
#include "Ayanami/Ayanami.SharedConst.h"

layout(local_size_x = kAyanamiGlobalDFRayMarchTileSize, local_size_y = kAyanamiGlobalDFRayMarchTileSize, local_size_z = 1) in;

RegisterUniform(bPerframe,{
    PerFramePerViewData data;
});

layout(push_constant) uniform PushConstant{
    vec4 m_GlobalDFBoxMin;
    vec4 m_GlobalDFBoxMax;
    uint m_PerFrameId;
    uint m_GlobalDFId;
    uint m_OutTex;
    uint m_RtH;
    uint m_RtW;
} pc;

bool rayboxIntersection(vec3 o,vec3 d,vec3 lb,vec3 rt, out float t){
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

void main(){
    float fov = GetResource(bPerframe, pc.m_PerFrameId).data.m_cameraFovX;
    float aspect = GetResource(bPerframe, pc.m_PerFrameId).data.m_cameraAspect;
    vec3 camPos = GetResource(bPerframe, pc.m_PerFrameId).data.m_cameraPosition.xyz;
    int tX = int(gl_GlobalInvocationID.x);
    int tY = int(gl_GlobalInvocationID.y);
    if(tX >= pc.m_RtW || tY >= pc.m_RtH) return;

    float ndcX = (2.0 * (float(tX)+0.5) / float(pc.m_RtW) - 1.0) * aspect;
    float ndcY = 1.0 - 2.0 * (float(tY)+0.5) / float(pc.m_RtH);
    float tanFov = tan(fov * 0.5);
    vec3 rayDir = normalize(vec3(ndcX * tanFov, ndcY * tanFov, 1.0));

    vec3 normal = vec3(0.0, 0.0, 0.0);
    float bestT = 1000000.0;

    // Ray-march the global DF
    vec3 rayOrigin = camPos;
    vec3 lb = pc.m_GlobalDFBoxMin.xyz;
    vec3 rt = pc.m_GlobalDFBoxMax.xyz;

    float t;
    bool hit = rayboxIntersection(rayOrigin, rayDir, lb, rt, t);
    vec3 normalEps = vec3(0.02, 0.02, 0.02);
    bool found = false;
    

    if(hit){
        for(int i=0;i<400;i++){
            vec3 p = rayOrigin + rayDir*t;
            vec3 uvw = (p - lb) / (rt - lb);
            uvw = clamp(uvw, vec3(0.0), vec3(1.0));
            float sdf = texture(GetSampler3D(pc.m_GlobalDFId), uvw).r;

            if(abs(sdf) < 0.001){
                found = true;
                float dx1 = texture(GetSampler3D(pc.m_GlobalDFId), uvw + vec3(normalEps.x, 0.0, 0.0)).r;
                float dx2 = texture(GetSampler3D(pc.m_GlobalDFId), uvw - vec3(normalEps.x, 0.0, 0.0)).r;
                float dy1 = texture(GetSampler3D(pc.m_GlobalDFId), uvw + vec3(0.0, normalEps.y, 0.0)).r;
                float dy2 = texture(GetSampler3D(pc.m_GlobalDFId), uvw - vec3(0.0, normalEps.y, 0.0)).r;
                float dz1 = texture(GetSampler3D(pc.m_GlobalDFId), uvw + vec3(0.0, 0.0, normalEps.z)).r;
                float dz2 = texture(GetSampler3D(pc.m_GlobalDFId), uvw - vec3(0.0, 0.0, normalEps.z)).r;

                normal.x = (dx1 - dx2);
                normal.y = (dy1 - dy2);
                normal.z = (dz1 - dz2);

                normal = normalize(normal);
                break;
            }

            t += sdf;
        }
    }
    
    imageStore(GetUAVImage2DR32F(pc.m_OutTex), ivec2(tX, tY), vec4(abs(normal),1.0));
}