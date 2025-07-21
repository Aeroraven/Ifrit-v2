
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


#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "SamplerUtils.SharedConst.h"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"



layout(
    local_size_x = kAyanamiGlobalDFRayMarchTileSize, 
    local_size_y = kAyanamiGlobalDFRayMarchTileSize, 
    local_size_z = 1) 
in;

RegisterStorage(BPerframe,{
    PerFramePerViewData m_Data;
});

RegisterStorage(BMeshDFDesc,{
    MeshDFDesc m_Data[];
});

RegisterStorage(BMeshDFMeta,{
    MeshDFMeta m_Data;
});

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});


layout(push_constant) uniform UPushConstant{
    vec4 m_GlobalDFBoxMin;
    vec4 m_GlobalDFBoxMax;
    uint m_PerFrameId;
    uint m_GlobalDFId;
    uint m_OutTex;
    uint m_RtH;
    uint m_RtW;
    uint m_GlobalObjectGridUAV;
    uint m_GlobalDFResolution;
    uint m_VoxelsPerClipMapWidth;
    uint m_MeshDFDescListId;
    uint m_AllCardData;
    uint m_CardResolution;
    uint m_CardAtlasResolution;
    uint m_CardDepthAtlasSRV;
    uint m_CardAlbedoAtlasSRV;
    uint m_CardDirectLightingAtlasSRV;
} PushConst;


void main(){
    float Fov = GetResource(BPerframe, PushConst.m_PerFrameId).m_Data.m_cameraFovX;
    float Aspect = GetResource(BPerframe, PushConst.m_PerFrameId).m_Data.m_cameraAspect;
    vec3 CamPos = GetResource(BPerframe, PushConst.m_PerFrameId).m_Data.m_cameraPosition.xyz;
    vec3 CamFront = GetResource(BPerframe, PushConst.m_PerFrameId).m_Data.m_cameraFront.xyz;

    vec3 zAxis = normalize(CamFront);
    vec3 xAxis = normalize(cross(vec3(0.0, 1.0, 0.0), zAxis));
    vec3 yAxis = normalize(cross(zAxis, xAxis));
    mat3 Rotation = mat3(xAxis, yAxis, zAxis);


    int tX = int(gl_GlobalInvocationID.x);
    int tY = int(gl_GlobalInvocationID.y);
    if(tX >= PushConst.m_RtW || tY >= PushConst.m_RtH) return;

    float NdcX = -(2.0 * (float(tX)+0.5) / float(PushConst.m_RtW) - 1.0) * Aspect;
    float NdcY = 1.0 - 2.0 * (float(tY)+0.5) / float(PushConst.m_RtH);
    float TanFov = tan(Fov * 0.5);
    vec3 RayDir = normalize(vec3(NdcX * TanFov, NdcY * TanFov, 1.0));
    RayDir = normalize(Rotation * RayDir);

    // Ray-march the global DF
    vec3 RayOrigin = CamPos;
    float HitTime = AyaShared_RayMarchGlobalDF(RayOrigin, RayDir, PushConst.m_GlobalDFId, PushConst.m_GlobalDFBoxMin.xyz,
        PushConst.m_GlobalDFBoxMax.xyz,0.005,0.005,200);
    
    // Get object grids from the global df grid
    if(HitTime<1e-3){
        imageStore(GetUAVImage2DR32F(PushConst.m_OutTex), ivec2(tX, tY), vec4(0.1,0.1,0.1, 1.0));
        return;
    }
    CardSample HitSample = AyaShared_EvaluateGlobalDFHit(RayOrigin, RayDir, HitTime, PushConst.m_GlobalDFId,
        PushConst.m_GlobalDFBoxMin.xyz, PushConst.m_GlobalDFBoxMax.xyz, PushConst.m_GlobalDFResolution,
        PushConst.m_VoxelsPerClipMapWidth, PushConst.m_GlobalObjectGridUAV, PushConst.m_MeshDFDescListId,
        PushConst.m_AllCardData,PushConst.m_CardDepthAtlasSRV,PushConst.m_CardAlbedoAtlasSRV,
        PushConst.m_CardResolution,PushConst.m_CardAtlasResolution,kAyanamiObjectGridTileSize);
    //HitSample.m_Albedo.xyz += 0.1;
    imageStore(GetUAVImage2DR32F(PushConst.m_OutTex), ivec2(tX, tY), vec4(HitSample.m_Albedo.xyz, 1.0));
}