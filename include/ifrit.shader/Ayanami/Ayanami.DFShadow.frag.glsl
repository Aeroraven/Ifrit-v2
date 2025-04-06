
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

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform UPushConstant{
    mat4 m_LightVP;
    vec4 m_LightDir;
    uint m_TileDFAtomics;
    uint m_TileDFList;
    uint m_TotalDFCount;
    uint m_TileSize;
    uint m_PerFrameId;
    uint m_DepthSRV;
    uint m_MeshDFDescListId;
    float m_ShadowCoefK; // This controls DFSS softness.
} PushConst;

RegisterStorage(BMeshDFDesc,{
    MeshDFDesc m_Data[];
});

RegisterStorage(BMeshDFMeta,{
    MeshDFMeta m_Data;
});

RegisterStorage(BTileAtomics,{
    uint m_Data[];
});

RegisterStorage(BTileScatter,{
    uint m_Data[];
});

RegisterUniform(BLocalTransform,{
    mat4 m_localToWorld;
    mat4 m_worldToLocal;
    vec4 m_maxScale;
});

RegisterUniform(BPerFrameData,{
    PerFramePerViewData data;
});


float RayMarchingForObject(uint meshDFId, vec3 rayOriginWS){
    MeshDFDesc desc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescListId).m_Data[meshDFId];
    MeshDFMeta meta = GetResource(BMeshDFMeta, desc.m_MdfMetaId).m_Data;
    mat4 worldToLocal = GetResource(BLocalTransform, desc.m_TransformId).m_worldToLocal;

    uint sdfId = meta.sdfId;
    vec3 lb = meta.bboxMin.xyz;
    vec3 rt = meta.bboxMax.xyz;
    vec3 extent = rt - lb;
    float maxExtent = min(min(extent.x, extent.y), extent.z);
    vec3 rayDir = normalize(-PushConst.m_LightDir.xyz);

    float advance = 0.01;
    vec4 pO = vec4(rayOriginWS + rayDir * advance, 1.0);
    pO = worldToLocal * pO;
    pO = pO / pO.w;

    vec4 pD = vec4(rayOriginWS + rayDir * (1.0+advance), 1.0);
    pD = worldToLocal * pD;
    pD = pD / pD.w;

    vec3 d = pD.xyz - pO.xyz;
    vec3 o = pO.xyz;
    vec3 nD = normalize(d);

    float t;
    bool hit = ifrit_RayboxIntersection(o,nD,lb,rt,t);
    t = max(0.0,t);

    vec3 hitp = o + nD * t;
    float retShadow = 1.0;
    float selfBias = 0e-4*maxExtent;
    float volBias = 1e-3*maxExtent;
    if(hit){
        for(int i=0;i<80;i++){
            vec3 uvw= (hitp - lb) / (rt - lb);
            uvw = clamp(uvw, 0.0, 1.0);
            float sdf = texture(GetSampler3D(meta.sdfId), uvw).x-volBias;
            t+= max(1e-4*maxExtent,abs(sdf)* 0.5) ;
            hitp = o + nD * t;
            
            retShadow = min(retShadow, PushConst.m_ShadowCoefK*abs(sdf)/(abs(t)+1e-6)*100.0);
            // if(abs(sdf)<2e-1){
            //     break;
            // }
        }
    }
    return retShadow;
}

float RayMarchingForAllGrid(uint gridId, vec3 rayOriginWS){
    float shadowAttn = 1.0;
    uint tileOffset = PushConst.m_TotalDFCount * gridId;
    uint dfInTile = GetResource(BTileAtomics, PushConst.m_TileDFAtomics).m_Data[gridId];
    for(uint i=0;i<PushConst.m_TotalDFCount;i++){
        shadowAttn = min(shadowAttn,RayMarchingForObject(i, rayOriginWS));
    }
    return shadowAttn;
}

float RayMarchingForGrid(uint gridId, vec3 rayOriginWS){
    float shadowAttn = 1.0;
    uint tileOffset = PushConst.m_TotalDFCount * gridId;
    uint dfInTile = GetResource(BTileAtomics, PushConst.m_TileDFAtomics).m_Data[gridId];
    for(uint i=0;i<dfInTile;i++){
        uint dfId = GetResource(BTileScatter, PushConst.m_TileDFList).m_Data[tileOffset + i];
        shadowAttn = min(shadowAttn,RayMarchingForObject(dfId, rayOriginWS));
    }
    return shadowAttn;
}

float RayMarchingFromWS(vec3 rayOriginWS){
    vec4 lightPos = PushConst.m_LightVP * vec4(rayOriginWS, 1.0);
    vec2 uv = (lightPos.xy / lightPos.w) * 0.5 + 0.5;

    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0){
        return 1.0;
    }

    uint tileX = uint(uv.x * PushConst.m_TileSize);
    uint tileY = uint(uv.y * PushConst.m_TileSize);
    uint tileId = tileX + tileY * PushConst.m_TileSize;
    
    return RayMarchingForGrid(tileId, rayOriginWS);
}

void main(){
    vec2 ndcXY = texCoord * 2.0 - 1.0;
    float depth = texture(GetSampler2D(PushConst.m_DepthSRV), texCoord).x;
    float camNear = GetResource(BPerFrameData, PushConst.m_PerFrameId).data.m_cameraNear;
    float camFar = GetResource(BPerFrameData, PushConst.m_PerFrameId).data.m_cameraFar;
    mat4 clipToWorld = GetResource(BPerFrameData, PushConst.m_PerFrameId).data.m_clipToWorld;
    float wsZ = ifrit_clipZToViewZ(depth, camNear, camFar);
    vec4 ndc = vec4(ndcXY, depth, 1.0) * wsZ;

    vec4 worldPos = clipToWorld * ndc;
    worldPos = worldPos / worldPos.w; 

    float shadowAttn = RayMarchingFromWS(worldPos.xyz);
    outColor = vec4(vec3(shadowAttn), 1.0); 
}