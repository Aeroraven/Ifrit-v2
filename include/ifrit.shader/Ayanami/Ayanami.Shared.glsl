
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




struct MeshDFDesc{
    uint m_MdfMetaId;
    uint m_TransformId;
};

struct MeshDFMeta{
    vec4 bboxMin;
    vec4 bboxMax;
    uint sdfId;
    uint m_IsTwoSided;
};

struct CardData{
    mat4 m_VP;
    mat4 m_VPInv;
};

struct CardAccumulator{
    vec4 m_AccAlbedo;
    vec4 m_MaxAlbedo;
    float m_Samples;
    float m_MaxWeight;
};

struct CardSample{
    vec4 m_Albedo;
};

RegisterStorage(BAyaShared_AllCardData,{
    CardData m_Mats[];
});


RegisterStorage(BAyaShared_MeshDFDesc,{
    MeshDFDesc m_Data[];
});

RegisterStorage(BAyaShared_MeshDFMeta,{
    MeshDFMeta m_Data;
});

RegisterUniform(BAyaShared_LocalTransform,{
    mat4 m_LocalToWorld;
    mat4 m_WorldToLocal;
    vec4 m_MaxScale;
});


RegisterStorage(BAyaShared_ObjectCell,{
    uvec4 m_Cell[];
});


vec2 AyaShared_GetSdfQuantScale(MeshDFMeta meta){
    float scaleMax = meta.bboxMax.w;
    float scaleMin = meta.bboxMin.w;
    return vec2(scaleMin,scaleMax);
}

float AyaShared_SampleMeshDF(uint SdfCombinedSRV,vec3 uvw, vec2 scale){
    float v = texture(GetSampler3D(SdfCombinedSRV),uvw).r;
    return mix(scale.x,scale.y,v);
}

float AyaShared_RayMarchGlobalDF(vec3 RayOrigin, vec3 RayDir, uint GlobalDFId, vec3 BoxMin, vec3 BoxMax, 
    float AcceptThreshold, float SdfExpansion, uint MaxSteps){

    float t,tMax;
    float HitTime = 0.0;
    bool Hit = ifrit_RayboxIntersectionDual(RayOrigin, RayDir, BoxMin, BoxMax, t,tMax);

    t = max(t, 0.0);
    vec3 InvBox = 1.0 / (BoxMax - BoxMin);
    if(Hit){
        for(int i=0;i<MaxSteps;i++){
            vec3 p = RayOrigin + RayDir*t;
            vec3 SdfUV = (p - BoxMin) * InvBox;
            uint SamplerId = sLinearClamp;
            float SdfVal = SampleTexture3D(GlobalDFId, SamplerId, SdfUV).r;
            SdfVal = SdfVal - SdfExpansion;
            if(SdfVal < AcceptThreshold){
                HitTime = t;
                break;
            }
            t += max(1e-2,SdfVal * 0.5);
            if(t>=tMax){
                break;
            }
        }
    }
    return HitTime;
}

void AyaShared_SampleCard(uint MeshId, uint CardFace, vec3 HitPosWS, vec3 HitNormalWS, vec3 HitPosMS, vec3 HitNormalMS,
    vec3 InNormalWeights, vec3 MeshExtent, uint AllCardData , uint DepthAtlasSRV, uint DirectLightingAtlasSRV,
    uint CardResolution,uint CardAtlasResolution, inout CardAccumulator Accum){

    uint CardId = MeshId * 6 + CardFace;
    CardData CardFaceData = GetResource(BAyaShared_AllCardData, AllCardData).m_Mats[CardId];
    mat4 ModelToCardScreen = CardFaceData.m_VP;
    vec3 HitPosCS = (ModelToCardScreen * vec4(HitPosMS, 1.0)).xyz;
    float ClampThreshold = 0.05;
    vec2 HitPosUV = HitPosCS.xy * 0.5 + 0.5;
    if(HitPosUV.x > -ClampThreshold && HitPosUV.x < (1.0+ClampThreshold) && HitPosUV.y > -ClampThreshold && HitPosUV.y < (1.0+ClampThreshold)){
        HitPosUV = clamp(HitPosUV, vec2(0.0), vec2(1.0));
    }
    if(HitPosUV.x < 0.0 || HitPosUV.x > 1.0 || HitPosUV.y < 0.0 || HitPosUV.y > 1.0){
        return;
    }
    
    uint CardsPerRow = CardAtlasResolution / CardResolution;
    uint CardIdX = CardId % CardsPerRow;
    uint CardIdY = CardId / CardsPerRow;
    uint CardOffsetX = CardIdX * CardResolution;
    uint CardOffsetY = CardIdY * CardResolution;
    uvec2 CardOffset = uvec2(CardOffsetX, CardOffsetY);
    uvec2 TileOffset = uvec2(HitPosUV * vec2(CardResolution));

    uvec2 AtlasOffset = CardOffset + TileOffset;
    vec2 AtlasUV = (vec2(AtlasOffset) + vec2(0.5)) / vec2(CardAtlasResolution);

    float ExtentZ = 1.0;
    float NormalWeights = 1.0;
    
    if(CardFace < 2){
        NormalWeights = InNormalWeights.x;
        ExtentZ = MeshExtent.x;
    }else if(CardFace < 4){
        NormalWeights = InNormalWeights.y;
        ExtentZ = MeshExtent.y;
    }else{
        NormalWeights = InNormalWeights.z;
        ExtentZ = MeshExtent.z;
    }

    float BiasOffset = 10.0/ExtentZ;
    float BiasFalloff = 0.25*BiasOffset;

    float CardDepth = SampleTexture2D(DepthAtlasSRV, sLinearClamp, AtlasUV).r;
    float TexelVisibility = 1.0;
    if(CardDepth >= 1.0){
        TexelVisibility = 0.0;
    }else{
        float HitDepth = HitPosCS.z;
        float HitDifference = (abs(HitDepth - CardDepth) - BiasOffset);
        HitDifference = clamp(HitDifference, 0.0, 1.0);
        TexelVisibility = 1.0-HitDifference;
    }

    float OverallWeights = NormalWeights * TexelVisibility;

    if(OverallWeights > 0.0){
        vec4 Albedo = SampleTexture2D(DirectLightingAtlasSRV, sLinearClamp, AtlasUV);
        Accum.m_AccAlbedo += Albedo * OverallWeights;
        Accum.m_Samples += OverallWeights;
        if(OverallWeights > Accum.m_MaxWeight){
            Accum.m_MaxWeight = OverallWeights;
            Accum.m_MaxAlbedo = Albedo;
        }
    }
}


void AyaShared_SampleMeshCards(uint MeshId, vec3 HitPosWS, vec3 HitNormalWS, uint MeshDFDescList, uint AllCardData,
    uint CardDepthAtlasSRV,uint CardAlbedoAtlasSRV, uint CardResolution, uint CardAtlasResolution,
    inout CardAccumulator Accum){

    const float kEPS = 1e-6;

    MeshDFDesc MeshDesc = GetResource(BAyaShared_MeshDFDesc, MeshDFDescList).m_Data[MeshId];
    uint TransformId = MeshDesc.m_TransformId;
    uint MdfMetaId = MeshDesc.m_MdfMetaId;
    MeshDFMeta MeshMeta = GetResource(BAyaShared_MeshDFMeta, MdfMetaId).m_Data;
    bool IsTwoSided = MeshMeta.m_IsTwoSided != 0;
    mat4 WorldToLocal = GetResource(BAyaShared_LocalTransform, TransformId).m_WorldToLocal;

    vec3 HitPosMS = (WorldToLocal * vec4(HitPosWS, 1.0)).xyz;
    vec3 HitNormalMS = (WorldToLocal * vec4(HitNormalWS, 0.0)).xyz;
    HitNormalMS = normalize(HitNormalMS);

    bool AlwaysTwoSided = false;
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
    vec3 BboxMax = MeshMeta.bboxMax.xyz;
    vec3 BboxMin = MeshMeta.bboxMin.xyz;
    vec3 MeshExtent = BboxMax - BboxMin;
    while(ValidOrientationMask>0){
        uint LowBit = findLSB(ValidOrientationMask);
        
        uint CardFace = LowBit;
        ValidOrientationMask &= ~(1 << LowBit);
        vec3 InNormalWeights = HitNormalMSSq;
        AyaShared_SampleCard(MeshId, CardFace, HitPosWS, HitNormalWS, HitPosMS, HitNormalMS, InNormalWeights,MeshExtent, 
            AllCardData, CardDepthAtlasSRV,CardAlbedoAtlasSRV,CardResolution,CardAtlasResolution, Accum);
    }
}

vec3 AyaShared_GetGlobalDistanceGradient(vec3 PosUVW, uint GlobalDFId, uint GlobalDFResolution){
    vec3 NormalEps = vec3(0.1/GlobalDFResolution);
    float dx1 = SampleTexture3D(GlobalDFId, sLinearClamp, PosUVW + vec3(NormalEps.x, 0.0, 0.0)).r;
    float dx2 = SampleTexture3D(GlobalDFId, sLinearClamp, PosUVW - vec3(NormalEps.x, 0.0, 0.0)).r;
    float dy1 = SampleTexture3D(GlobalDFId, sLinearClamp, PosUVW + vec3(0.0, NormalEps.y, 0.0)).r;
    float dy2 = SampleTexture3D(GlobalDFId, sLinearClamp, PosUVW - vec3(0.0, NormalEps.y, 0.0)).r;
    float dz1 = SampleTexture3D(GlobalDFId, sLinearClamp, PosUVW + vec3(0.0, 0.0, NormalEps.z)).r;
    float dz2 = SampleTexture3D(GlobalDFId, sLinearClamp, PosUVW - vec3(0.0, 0.0, NormalEps.z)).r;
    return normalize(vec3(dx1 - dx2, dy1 - dy2, dz1 - dz2));
}

vec3 AyaShared_GetGlobalDistanceUVWFromWorldPos(vec3 WorldPos, vec3 GlobalDFMin, vec3 GlobalDFMax){
    vec3 BoxMin = GlobalDFMin;
    vec3 BoxMax = GlobalDFMax;
    vec3 SdfUV = (WorldPos - BoxMin) / (BoxMax - BoxMin);
    SdfUV = clamp(SdfUV, vec3(0.0), vec3(1.0));
    return SdfUV;
}

uint AyaShared_GetObjectGridIdFromHitPos(vec3 WorldPos,vec3 GlobalDFMin, vec3 GlobalDFMax,uint VoxelsPerClipMapWidth){
    vec3 SdfUV = AyaShared_GetGlobalDistanceUVWFromWorldPos(WorldPos,GlobalDFMin,GlobalDFMax);
    uvec3 ObjGrid = uvec3(SdfUV * vec3(VoxelsPerClipMapWidth));
    uint CellLoc = ifrit_ToCellId(ObjGrid, uvec3(VoxelsPerClipMapWidth));
    return CellLoc;
}


CardSample AyaShared_EvaluateGlobalDFHit(vec3 TraceOriginWS, vec3 TraceDirWS, float HitTime, uint GlobalDFSRV,
    vec3 GlobalDFBoxMin, vec3 GlobalDFBoxMax, uint GlobalDFResolution, uint VoxelsPerClipMapwidth,
    uint ObjectGridUAV, uint MeshDFDescListId, uint AllCardData, uint CardDepthAtlasSRV,uint CardAlbedoAtlasSRV,
    uint CardResolution, uint CardAtlasResolution, uint NumGridTileObjects){

    CardSample Sample;
    Sample.m_Albedo = vec4(0.0);

    CardAccumulator Accum;
    Accum.m_AccAlbedo = vec4(0.0);
    Accum.m_MaxAlbedo = vec4(0.0);
    Accum.m_Samples = 0.0;
    Accum.m_MaxWeight = 0.0;

    vec3 HitPosWS = TraceOriginWS + TraceDirWS * HitTime;
    vec3 HitPosSdfUVW = AyaShared_GetGlobalDistanceUVWFromWorldPos(HitPosWS,GlobalDFBoxMin,GlobalDFBoxMax);
    vec3 HitNormalWS = AyaShared_GetGlobalDistanceGradient(HitPosSdfUVW,GlobalDFSRV,GlobalDFResolution);
    uint ObjGridId = AyaShared_GetObjectGridIdFromHitPos(HitPosWS,GlobalDFBoxMin,GlobalDFBoxMax,VoxelsPerClipMapwidth);

    uvec4 ObjGridData = GetResource(BAyaShared_ObjectCell, ObjectGridUAV).m_Cell[ObjGridId];
    float ValidSamples=0.0;
    for(uint ObjSlotId=0;ObjSlotId<NumGridTileObjects;ObjSlotId++){
        uint ObjGridSlotData = ObjGridData[ObjSlotId];
        uint MeshId = ObjGridSlotData & 0xFFFFFF;
        uint DepthId = (ObjGridSlotData >> 24) & 0xFF;
        if(DepthId == 0xFF) continue;

        // Got it, sample its cards!
        AyaShared_SampleMeshCards(MeshId, HitPosWS, HitNormalWS,MeshDFDescListId,AllCardData, 
            CardDepthAtlasSRV, CardAlbedoAtlasSRV, CardResolution,CardAtlasResolution, Accum);
        ValidSamples += 1.0;
    }

    if(Accum.m_Samples > 0.0){
        Sample.m_Albedo =  Accum.m_AccAlbedo / Accum.m_Samples;
    }else{
        Sample.m_Albedo = vec4(0.05, 0.05, 0.05, 1.0);
    }
    return Sample;
}