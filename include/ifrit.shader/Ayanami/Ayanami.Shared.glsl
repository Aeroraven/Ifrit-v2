
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

struct RadiosityRayCardSample{
    vec3 m_WorldPos;
    vec3 m_WorldNormal;
    bool m_ValidSample; // In atlas but hit nothing
    bool m_PresentInAtlas;
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

RegisterUniform(BAyaShared_Perframe,{
    PerFramePerViewData data;
});


// ==== Common ====
PerFramePerViewData AyaShared_GetPerFrameData(uint PerFrameId){
    return GetResource(BAyaShared_Perframe, PerFrameId).data;
}
MeshDFMeta AyaShared_GetMeshDFData(uint MeshDescId,uint MDFId){
    MeshDFDesc MeshDesc = GetResource(BAyaShared_MeshDFDesc, MeshDescId).m_Data[MDFId];
    MeshDFMeta MeshMeta = GetResource(BAyaShared_MeshDFMeta, MeshDesc.m_MdfMetaId).m_Data;
    return MeshMeta;
}

mat4 AyaShared_GetLocalToWorld(uint MeshDescId,uint MDFId){
    MeshDFDesc MeshDesc = GetResource(BAyaShared_MeshDFDesc, MeshDescId).m_Data[MDFId];
    uint TransformId = MeshDesc.m_TransformId;
    return GetResource(BAyaShared_LocalTransform, TransformId).m_LocalToWorld;
}

mat4 AyaShared_GetWorldToLocalMesh(uint MeshDescId,uint MDFId){
    MeshDFDesc MeshDesc = GetResource(BAyaShared_MeshDFDesc, MeshDescId).m_Data[MDFId];
    uint TransformId = MeshDesc.m_TransformId;
    return GetResource(BAyaShared_LocalTransform, TransformId).m_WorldToLocal;
}

vec3 AyaShared_GetMeshDFScale(uint MeshDescId,uint MDFId){
    MeshDFDesc MeshDesc = GetResource(BAyaShared_MeshDFDesc, MeshDescId).m_Data[MDFId];
    uint TransformId = MeshDesc.m_TransformId;
    vec4 Scale = GetResource(BAyaShared_LocalTransform, TransformId).m_MaxScale;
    return Scale.xyz;
}

// ====
vec4 AyaShared_GetWorldPosFromDepthPersp(PerFramePerViewData PerFrame,float Depth, vec2 UV){
    vec3 NdcXyz = vec3(UV*2.0-1.0, Depth);
    float CamNear = PerFrame.m_cameraNear;
    float CamFar = PerFrame.m_cameraFar;
    float WorldDepth = ifrit_recoverViewSpaceDepth(Depth, CamNear, CamFar);
    vec4 Ndc = vec4(NdcXyz, 1.0) * WorldDepth;
    mat4 ClipToWorld = PerFrame.m_clipToWorld;
    vec4 WorldPos = ClipToWorld * Ndc;
    return vec4(WorldPos.xyz / WorldPos.w, WorldDepth);
}

vec2 AyaShared_GetSdfQuantScale(MeshDFMeta meta){
    float scaleMax = meta.bboxMax.w;
    float scaleMin = meta.bboxMin.w;
    return vec2(scaleMin,scaleMax);
}

float AyaShared_SampleMeshDF(uint SdfCombinedSRV,vec3 uvw, vec2 scale){
    float v = SampleTexture3D(SdfCombinedSRV,sLinearClamp,uvw).r;
    return mix(scale.x,scale.y,v);
}

float AyaShared_RayMarchGlobalDF(vec3 RayOrigin, vec3 RayDir, uint GlobalDFId, vec3 BoxMin, vec3 BoxMax, 
    float AcceptThreshold, float SdfExpansion, uint MaxSteps){

    float t,tMax;
    float HitTime = -1.0;
    bool Hit = ifrit_RayboxIntersectionDual(RayOrigin, RayDir, BoxMin, BoxMax, t,tMax);

    t = max(t, 1e-3);
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
            t += max(1e-4,SdfVal * 0.5);
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

    float BiasOffset = 0.01/ExtentZ;
    float BiasFalloff = 0.25*BiasOffset;

    float CardDepth = SampleTexture2D(DepthAtlasSRV, sLinearClamp, AtlasUV).r;
    float TexelVisibility = 1.0;
    if(CardDepth >= 1.0){
        TexelVisibility = 0.0;
    }else{
        float HitDepth = HitPosCS.z;
        float HitDifference = (abs(HitDepth - CardDepth) - BiasOffset)/BiasFalloff;
        HitDifference = clamp(HitDifference, 0.0, 1.0);
        TexelVisibility = 1.0-HitDifference;
    }

    float OverallWeights = NormalWeights * TexelVisibility;

    if(OverallWeights >= 0.0){
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

    if(Accum.m_MaxWeight >= 0.97){
        Sample.m_Albedo = Accum.m_MaxAlbedo;
    }
    
    return Sample;
}

// ==== Radiosity Tracing ====

void AyaShared_RayTraceCoordToCardInfo(uint GThreadId, vec2 Jitter, out uvec2 OffsetInCardTile,out uint CardTileId, out uvec2 TraceRayCoord){
    uint ProbeId = GThreadId % kAyanami_RadiosityTracesPerProbe;
    
    uint ProbesPerTile = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityProbesPerCardTileWidth;
    uint CurTileId = ProbeId / ProbesPerTile;
    uint CurProbeIdInTile = GThreadId % ProbesPerTile;
    uvec2 ProbePosInTile = uvec2(CurProbeIdInTile % kAyanami_RadiosityProbesPerCardTileWidth,
                                 CurProbeIdInTile / kAyanami_RadiosityProbesPerCardTileWidth);

    uint ProbeSpacing = kAyanami_CardTileWidth / kAyanami_RadiosityProbesPerCardTileWidth;
    uvec2 ProbeOffset = ProbePosInTile * ProbeSpacing + uvec2(ProbeSpacing * Jitter);
    OffsetInCardTile = ProbeOffset;
    CardTileId = CurTileId;

    uint RayOffsetInProbe =  GThreadId % kAyanami_RadiosityTracesPerProbe;
    uvec2 RayCoord = uvec2(RayOffsetInProbe % kAyanami_RadiosityProbHemiRes,
                             RayOffsetInProbe / kAyanami_RadiosityProbHemiRes);
    TraceRayCoord = RayCoord;
}

// Atlas = 8192x8192
// Tile = 8x8
// 2x2 Probes per tile => 1 probe = 4x4 area (16traces) => 1 probe = 16 storage slots for radiance
uvec2 AyaShared_GetRadianceSlot(uint TileIndex, uvec2 OffsetInTile, uvec2 TraceRayCoord, uint CardAtlasResolution){
    uint TilesPerAtlasWidth = CardAtlasResolution / kAyanami_CardTileWidth;
    uint TileX = TileIndex % TilesPerAtlasWidth;
    uint TileY = TileIndex / TilesPerAtlasWidth;
    uvec2 OffsetByTile = uvec2(TileX, TileY) * kAyanami_CardTileWidth;

    uvec2 InTileProbeId = OffsetInTile / kAyanami_RadiosityProbHemiRes;
    uvec2 OffsetByProbe = InTileProbeId * kAyanami_RadiosityProbHemiRes;
    return OffsetByTile + OffsetInTile + TraceRayCoord;
}


mat4 AyaShared_GetCardMeshLocalToWorld(uint CardId, uint AllMeshDFDataId){
    uint MeshDFId = CardId / 6;
    MeshDFDesc MeshDesc = GetResource(BAyaShared_MeshDFDesc, AllMeshDFDataId).m_Data[MeshDFId];
    MeshDFMeta MeshMeta = GetResource(BAyaShared_MeshDFMeta, AllMeshDFDataId).m_Data;
    uint TransformId = MeshDesc.m_TransformId;
    return GetResource(BAyaShared_LocalTransform, TransformId).m_LocalToWorld;
}

mat4 AyaShared_GetCardViewVPToWorld(uint CardId, uint AllCardObjDataId){
    CardData CardDesc = GetResource(BAyaShared_AllCardData, AllCardObjDataId).m_Mats[CardId];
    return CardDesc.m_VPInv;
}

RadiosityRayCardSample AyaShared_RadiosityRayCardSample(uint CardId, uvec2 InCardUV, uvec2 AtlasUV, uint CardResolution, uint CardAtlasResolution,
    uint CardDepthAtlasSRV, uint CardNormalAtlasSRV, uint AllCardObjDataId, uint AllMeshDFDataId){

    vec2 InCardUVF = (vec2(InCardUV) + vec2(0.5)) / float(CardResolution);
    vec2 AtlasUVF = (vec2(AtlasUV) + vec2(0.5)) / float(CardAtlasResolution);
    InCardUVF = InCardUVF * 2.0 - 1.0;
    AtlasUVF = AtlasUVF * 2.0 - 1.0;

    float Depth = SampleTexture2D(CardDepthAtlasSRV,sLinearClamp,AtlasUVF).r;
    vec2 LocalNormalRG = SampleTexture2D(CardNormalAtlasSRV,sLinearClamp,AtlasUVF).rg * 2.0 - 1.0;
    vec3 LocalNormal = normalize(vec3(LocalNormalRG, sqrt(1.0 - dot(LocalNormalRG, LocalNormalRG))));

    if(Depth == 1.0){
        RadiosityRayCardSample SampledData;
        SampledData.m_WorldPos = vec3(0.0);
        SampledData.m_WorldNormal = vec3(0.0);
        SampledData.m_ValidSample = false;
        return SampledData;
    }

    vec4 OrthoNDC = vec4(InCardUVF, Depth, 1.0);
    mat4 CardViewToLocal = AyaShared_GetCardViewVPToWorld(CardId, AllCardObjDataId);
    vec4 LocalPos = CardViewToLocal * OrthoNDC;
    LocalPos /= LocalPos.w;

    mat4 CardMeshToWorld = AyaShared_GetCardMeshLocalToWorld(CardId, AllMeshDFDataId);
    vec4 WorldNormal = CardMeshToWorld * vec4(LocalNormal, 0.0);
    vec4 WorldPos = CardMeshToWorld * vec4(LocalPos.xyz, 1.0);
    WorldPos /= WorldPos.w;
    WorldNormal = normalize(WorldNormal);

    RadiosityRayCardSample SampledData;
    SampledData.m_WorldPos = WorldPos.xyz;
    SampledData.m_WorldNormal = WorldNormal.xyz;
    SampledData.m_ValidSample = true;
}

RadiosityRayCardSample AyaShared_GetRadiosityRayCardSample(uint CardTileId, uvec2 OffsetInCardTile, uint CardAtlasResolution, 
    uint CardResolution, uint NumCards, uint CardDepthAtlasSRV, uint CardNormalAtlasSRV, uint AllCardObjDataId, uint AllMeshDFDataId){
        
    uint TilesPerAtlasWidth = CardAtlasResolution / kAyanami_CardTileWidth;
    uint TileX = CardTileId % TilesPerAtlasWidth;
    uint TileY = CardTileId / TilesPerAtlasWidth;

    uint CardX = TileX * kAyanami_CardTileWidth + OffsetInCardTile.x;
    uint CardY = TileY * kAyanami_CardTileWidth + OffsetInCardTile.y;
    uint CardIdX = CardX % CardResolution;
    uint CardIdY = CardY % CardResolution;
    uint CardId = CardIdX + CardIdY * CardResolution;

    uvec2 InCardUV = uvec2(CardX, CardY) % kAyanami_CardTileWidth;
    uvec2 AtlasUV = uvec2(CardX, CardY);

    if(CardId >= NumCards){
        RadiosityRayCardSample SampledData;
        SampledData.m_WorldPos = vec3(0.0);
        SampledData.m_WorldNormal = vec3(0.0);
        SampledData.m_ValidSample = false;
        SampledData.m_PresentInAtlas = false;
        return SampledData;
    }

    RadiosityRayCardSample SampledData = AyaShared_RadiosityRayCardSample(CardId, InCardUV, AtlasUV, CardResolution, CardAtlasResolution,
        CardDepthAtlasSRV, CardNormalAtlasSRV, AllCardObjDataId, AllMeshDFDataId);
    SampledData.m_PresentInAtlas = true;
    return SampledData;
}

