
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
#include "SamplerUtils.SharedConst.h"

layout(
    local_size_x = kAyanamiGlobalDFRayMarchTileSize, 
    local_size_y = kAyanamiGlobalDFRayMarchTileSize, 
    local_size_z = 1) 
in;

RegisterUniform(BPerframe,{
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


RegisterStorage(BObjectCell,{
    uvec4 m_Cell[];
});

RegisterUniform(BLocalTransform,{
    mat4 m_LocalToWorld;
    mat4 m_WorldToLocal;
    vec4 m_MaxScale;
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
} PushConst;

struct CardSample{
    vec4 m_Albedo;
};

struct CardAccumulator{
    vec4 m_AccAlbedo;
    vec4 m_MaxAlbedo;
    float m_Samples;
    float m_MaxWeight;
};

const float kEPS = 1e-4;

vec3 GetGlobalDistanceGradient(vec3 PosUVW){
    vec3 NormalEps = vec3(0.5/PushConst.m_GlobalDFResolution);
    float dx1 = SampleTexture3D(PushConst.m_GlobalDFId, sLinearClamp, PosUVW + vec3(NormalEps.x, 0.0, 0.0)).r;
    float dx2 = SampleTexture3D(PushConst.m_GlobalDFId, sLinearClamp, PosUVW - vec3(NormalEps.x, 0.0, 0.0)).r;
    float dy1 = SampleTexture3D(PushConst.m_GlobalDFId, sLinearClamp, PosUVW + vec3(0.0, NormalEps.y, 0.0)).r;
    float dy2 = SampleTexture3D(PushConst.m_GlobalDFId, sLinearClamp, PosUVW - vec3(0.0, NormalEps.y, 0.0)).r;
    float dz1 = SampleTexture3D(PushConst.m_GlobalDFId, sLinearClamp, PosUVW + vec3(0.0, 0.0, NormalEps.z)).r;
    float dz2 = SampleTexture3D(PushConst.m_GlobalDFId, sLinearClamp, PosUVW - vec3(0.0, 0.0, NormalEps.z)).r;
    return normalize(vec3(dx1 - dx2, dy1 - dy2, dz1 - dz2));
}

vec3 GetGlobalDistanceUVWFromWorldPos(vec3 WorldPos){
    vec3 BoxMin = PushConst.m_GlobalDFBoxMin.xyz;
    vec3 BoxMax = PushConst.m_GlobalDFBoxMax.xyz;
    vec3 SdfUV = (WorldPos - BoxMin) / (BoxMax - BoxMin);
    SdfUV = clamp(SdfUV, vec3(0.0), vec3(1.0));
    return SdfUV;
}

uint GetObjectGridIdFromHitPos(vec3 WorldPos){
    vec3 SdfUV = GetGlobalDistanceUVWFromWorldPos(WorldPos);
    uvec3 ObjGrid = uvec3(SdfUV * vec3(PushConst.m_VoxelsPerClipMapWidth));
    uint CellLoc = ifrit_ToCellId(ObjGrid, uvec3(PushConst.m_VoxelsPerClipMapWidth));
    return CellLoc;
}

void SampleCard(uint MeshId, uint CardFace, vec3 HitPosWS, vec3 HitNormalWS, vec3 HitPosMS, vec3 HitNormalMS,
    vec3 InNormalWeights, inout CardAccumulator Accum){

    uint CardId = MeshId * 6 + CardFace;
    CardData CardFaceData = GetResource(BAllCardData, PushConst.m_AllCardData).m_Mats[CardId];
    mat4 ModelToCardScreen = CardFaceData.m_VP;
    vec3 HitPosCS = (ModelToCardScreen * vec4(HitPosMS, 1.0)).xyz;

    vec2 HitPosUV = HitPosCS.xy * 0.5 + 0.5;
    if(HitPosUV.x < 0.0 || HitPosUV.x > 1.0 || HitPosUV.y < 0.0 || HitPosUV.y > 1.0){
        return;
    }
    
    uint CardsPerRow = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
    uint CardIdX = CardId % CardsPerRow;
    uint CardIdY = CardId / CardsPerRow;
    uint CardOffsetX = CardIdX * PushConst.m_CardResolution;
    uint CardOffsetY = CardIdY * PushConst.m_CardResolution;
    uvec2 CardOffset = uvec2(CardOffsetX, CardOffsetY);
    uvec2 TileOffset = uvec2(HitPosUV * vec2(PushConst.m_CardResolution));

    uvec2 AtlasOffset = CardOffset + TileOffset;
    vec2 AtlasUV = (vec2(AtlasOffset) + vec2(0.5)) / vec2(PushConst.m_CardAtlasResolution);

    float CardDepth = SampleTexture2D(PushConst.m_CardDepthAtlasSRV, sLinearClamp, AtlasUV).r;
    float TexelVisibility = 1.0;
    if(CardDepth >= 1.0){
        TexelVisibility = 0.0;
    }
    float NormalWeights = 1.0;
    if(CardFace < 2){
        NormalWeights = InNormalWeights.x;
    }else if(CardFace < 4){
        NormalWeights = InNormalWeights.y;
    }else{
        NormalWeights = InNormalWeights.z;
    }

    float OverallWeights = NormalWeights * TexelVisibility;

    if(OverallWeights > 0.0){
        vec4 Albedo = SampleTexture2D(PushConst.m_CardAlbedoAtlasSRV, sLinearClamp, AtlasUV);
        Accum.m_AccAlbedo += Albedo * OverallWeights;
        Accum.m_Samples += OverallWeights;
        if(OverallWeights > Accum.m_MaxWeight){
            Accum.m_MaxWeight = OverallWeights;
            Accum.m_MaxAlbedo = Albedo;
        }
    }

}

void SampleMesh(uint MeshId, vec3 HitPosWS, vec3 HitNormalWS,inout CardAccumulator Accum){
    MeshDFDesc MeshDesc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescListId).m_Data[MeshId];
    uint TransformId = MeshDesc.m_TransformId;
    mat4 WorldToLocal = GetResource(BLocalTransform, TransformId).m_WorldToLocal;

    vec3 HitPosMS = (WorldToLocal * vec4(HitPosWS, 1.0)).xyz;
    vec3 HitNormalMS = (WorldToLocal * vec4(HitNormalWS, 0.0)).xyz;
    HitNormalMS = normalize(HitNormalMS);

    vec3 HitNormalMSSq = HitNormalMS * HitNormalMS;
    // X->1,2; Y->3,4 Z->5,6
    uint SampleDirectionMask = 0;
    if(HitNormalMSSq.x>=kEPS){
        if(HitNormalMS.x<0.0){
            SampleDirectionMask |= 1;
        }else{
            SampleDirectionMask |= 2;
        }
    }
    if(HitNormalMSSq.y>=kEPS){
        if(HitNormalMS.y<0.0){
            SampleDirectionMask |= 4;
        }else{
            SampleDirectionMask |= 8;
        }
    }
    if(HitNormalMSSq.z>=kEPS){
        if(HitNormalMS.z<0.0){
            SampleDirectionMask |= 16;
        }else{
            SampleDirectionMask |= 32;
        }
    }

    uint ValidOrientationMask = SampleDirectionMask;
    while(ValidOrientationMask>0){
        uint LowBit = findLSB(ValidOrientationMask);
        
        uint CardFace = LowBit;
        ValidOrientationMask &= ~(1 << LowBit);
        vec3 InNormalWeights = HitNormalMSSq;
        SampleCard(MeshId, CardFace, HitPosWS, HitNormalWS, HitPosMS, HitNormalMS, InNormalWeights, Accum);
    }
}

CardSample EvaluateGlobalDFHit(vec3 TraceOriginWS, vec3 TraceDirWS, float HitTime){
    CardSample Sample;
    Sample.m_Albedo = vec4(0.0);

    CardAccumulator Accum;
    Accum.m_AccAlbedo = vec4(0.0);
    Accum.m_MaxAlbedo = vec4(0.0);
    Accum.m_Samples = 0.0;
    Accum.m_MaxWeight = 0.0;

    vec3 HitPosWS = TraceOriginWS + TraceDirWS * HitTime;
    vec3 HitPosSdfUVW = GetGlobalDistanceUVWFromWorldPos(HitPosWS);
    vec3 HitNormalWS = GetGlobalDistanceGradient(HitPosSdfUVW);
    uint ObjGridId = GetObjectGridIdFromHitPos(HitPosWS);

    uvec4 ObjGridData = GetResource(BObjectCell, PushConst.m_GlobalObjectGridUAV).m_Cell[ObjGridId];
    float ValidSamples=0.0;
    for(uint ObjSlotId=0;ObjSlotId<kAyanamiObjectGridTileSize;ObjSlotId++){
        uint ObjGridSlotData = ObjGridData[ObjSlotId];
        uint MeshId = ObjGridSlotData & 0xFFFFFF;
        uint DepthId = (ObjGridSlotData >> 24) & 0xFF;
        if(DepthId == 0xFF) continue;

        // Got it, sample its cards!
        SampleMesh(MeshId, HitPosWS, HitNormalWS, Accum);
        ValidSamples += 1.0;
    }

    if(Accum.m_Samples > -10.0){
        Sample.m_Albedo =  Accum.m_AccAlbedo / Accum.m_Samples;
    }

    return Sample;
}

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

    vec3 Normal = vec3(0.0, 0.0, 0.0);

    // Ray-march the global DF
    vec3 RayOrigin = CamPos;
    vec3 BoxMin = PushConst.m_GlobalDFBoxMin.xyz;
    vec3 BoxMax = PushConst.m_GlobalDFBoxMax.xyz;

    float t;
    float HitTime = 0.0;
    bool Hit = ifrit_RayboxIntersection(RayOrigin, RayDir, BoxMin, BoxMax, t);

    t = max(t, 0.0);
    vec3 NormalEps = vec3(0.5/PushConst.m_GlobalDFResolution);
    
    if(Hit){
        for(int i=0;i<200;i++){
            vec3 p = RayOrigin + RayDir*t;
            vec3 SdfUV = (p - BoxMin) / (BoxMax - BoxMin);
            SdfUV = clamp(SdfUV, vec3(0.0), vec3(1.0));
            float SdfVal = SampleTexture3D(PushConst.m_GlobalDFId, sLinearClamp, SdfUV).r - 0.0225;

            if(SdfVal < 0.0125){
                HitTime = t;
                Normal = GetGlobalDistanceGradient(SdfUV);
                break;
            }

            t += max(1e-2,SdfVal * 0.5);
        }
    }
    
    // Get object grids from the global df grid
    if(HitTime<1e-3){
        imageStore(GetUAVImage2DR32F(PushConst.m_OutTex), ivec2(tX, tY), vec4(1.0,0.0,0.0, 1.0));
        return;
    }
    CardSample HitSample = EvaluateGlobalDFHit(RayOrigin, RayDir, HitTime);
    //HitSample.m_Albedo.xyz += 0.1;
    imageStore(GetUAVImage2DR32F(PushConst.m_OutTex), ivec2(tX, tY), vec4(HitSample.m_Albedo.xyz, 1.0));
}