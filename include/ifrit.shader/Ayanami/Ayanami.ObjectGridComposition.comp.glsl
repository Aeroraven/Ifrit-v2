
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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
#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"

layout(
    local_size_x = kAyanamiObjectGridTileSize, 
    local_size_y = kAyanamiObjectGridTileSize, 
    local_size_z = kAyanamiObjectGridTileSize 
) in;

layout(push_constant) uniform UPushConstant{
    uint m_NumTotalMeshDF;
    uint m_MeshDFDescListId;
    float m_ClipMapRadius;
    uint m_VoxelsPerClipMapWidth;
    uint m_CellDataId;
} PushConst;

RegisterStorage(BMeshDFDesc,{
    MeshDFDesc m_Data[];
});

RegisterStorage(BMeshDFMeta,{
    MeshDFMeta m_Data;
});

RegisterUniform(BLocalTransform,{
    mat4 m_LocalToWorld;
    mat4 m_WorldToLocal;
    vec4 m_MaxScale;
});

RegisterStorage(BObjectCell,{
    uvec4 m_Cell[];
});

shared uint LocalSharedCullResult[kAyanami_ObjectGridCellMaxCullObjPerPass];
shared uint LocalSharedCullResultCount;

float ClosestDistanceToSDF(MeshDFMeta MdfMeta, vec3 QueryPos, vec3 MeshScale, mat4 WorldToLocal){
    uint SDFId = MdfMeta.sdfId;
    float MeshMaxScale = max(MeshScale.x, max(MeshScale.y, MeshScale.z));

    vec4 CellLocalCoord = WorldToLocal*vec4(QueryPos, 1.0);
    vec3 CellLocalCoord3 = CellLocalCoord.xyz/CellLocalCoord.w;

    vec3 MeshBBoxMin = MdfMeta.bboxMin.xyz;
    vec3 MeshBBoxMax = MdfMeta.bboxMax.xyz;
    vec3 MeshBBoxCenter = (MeshBBoxMin+MeshBBoxMax)*0.5;
    vec3 MeshBBoxExtent = (MeshBBoxMax-MeshBBoxMin)*0.5;

    vec3 CellLocalCoord3RelativeToCenter = CellLocalCoord3-MeshBBoxCenter;
    vec3 ToBox = (abs(CellLocalCoord3RelativeToCenter)-MeshBBoxExtent) * MeshScale;

    float ToBoxOut = length(max(ToBox, vec3(0.0)));
    float ToBoxIn = min(max(ToBox.x,max(ToBox.y,ToBox.z)), 0.0);
    float ToBoxAll = ToBoxOut + ToBoxIn;
    float ToBoxAllPositive = max(ToBoxAll, 0.0);


    vec3 ClampedPos = clamp(CellLocalCoord3RelativeToCenter, -MeshBBoxExtent, MeshBBoxExtent);
    ClampedPos = ClampedPos + MeshBBoxCenter;

    vec3 ClampedUVW = (ClampedPos-MeshBBoxMin)/(MeshBBoxMax-MeshBBoxMin);

    //float SdfVal = SampleTexture3D(SDFId,sLinearClamp,ClampedUVW).r * MeshMaxScale; 
    float SdfVal = texture(GetSampler3D(SDFId), ClampedUVW).r * MeshMaxScale;
    float TotalSdf = max(SdfVal + ToBoxAllPositive,ToBoxAll);
    return TotalSdf;
}

uint PackCellData(uint MeshId,float HitDist,float CellWidth){
    float HitDistRaw = HitDist / CellWidth;
    HitDistRaw = clamp(HitDistRaw, 0.0, 1.0);
    uint HitDistInt = uint( HitDistRaw * 0xFF) & 0xFF;
    return (MeshId & 0xFFFFFF) | (HitDistInt << 24);
}

void AddObjectToGridCell(uint MeshId, uint CellId, float HitDist,float CellWidth){
    uint MaxIndex = 0;
    uvec4 CellData = GetResource(BObjectCell, PushConst.m_CellDataId).m_Cell[CellId];
    uint PackedData = PackCellData(MeshId, HitDist, CellWidth);
    for(uint i=0;i<kAyanamiObjectGridTileSize;i++){
        MaxIndex = max(MaxIndex, CellData[i]);
    }
    for(uint i=0;i<kAyanamiObjectGridTileSize;i++){
        if(CellData[i] == MaxIndex){
            if(PackedData < CellData[i]){
                CellData[i] = PackedData;
                GetResource(BObjectCell, PushConst.m_CellDataId).m_Cell[CellId] = CellData;
                break;
            }
        }
    }
}

void SortGridCell(uint CellId){
    uvec4 CellData = GetResource(BObjectCell, PushConst.m_CellDataId).m_Cell[CellId];
    uint CellDataArr[kAyanamiObjectGridTileSize];
    for(uint i=0;i<kAyanamiObjectGridTileSize;i++){
        CellDataArr[i] = CellData[i];
    }
    for(uint i=0;i<kAyanamiObjectGridTileSize;i++){
        for(uint j=i+1;j<kAyanamiObjectGridTileSize;j++){
            if(CellDataArr[i] > CellDataArr[j]){
                uint Temp = CellDataArr[i];
                CellDataArr[i] = CellDataArr[j];
                CellDataArr[j] = Temp;
            }
        }
    }
    for(uint i=0;i<kAyanamiObjectGridTileSize;i++){
        CellData[i] = CellDataArr[i];
    }
    GetResource(BObjectCell, PushConst.m_CellDataId).m_Cell[CellId] = CellData;
}


// Almost no resources about this. So stuffs are all personal guess.
// I cannot ensure this is correct. >_<
void main(){
    uvec3 TileId = gl_WorkGroupID;
    uvec3 CellId = gl_GlobalInvocationID;
    uint LocalId = ifrit_ToCellId(uvec3(gl_LocalInvocationID),uvec3(gl_WorkGroupSize));
    uint LocalSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;

    uint NumCullingPasses = ifrit_DivRoundUp(PushConst.m_NumTotalMeshDF, kAyanami_ObjectGridCellMaxCullObjPerPass);
    uint NumObjectGridTiles = kAyanamiObjectGridTileSize / PushConst.m_VoxelsPerClipMapWidth;

    vec3 TileOffsetInCells = vec3(TileId) / NumObjectGridTiles;
    vec3 TileOffsetInCellsNext = vec3(TileId + 1) / NumObjectGridTiles;

    vec3 TileCenterCoord  = ((TileOffsetInCells + TileOffsetInCellsNext) * 0.5) * 2.0 - 1.0;
    TileCenterCoord *= PushConst.m_ClipMapRadius;
    vec3 TileExtent = (TileOffsetInCellsNext - TileOffsetInCells) * 2.0 * PushConst.m_ClipMapRadius;

    float CellWidth = PushConst.m_ClipMapRadius * 2.0 / float(PushConst.m_VoxelsPerClipMapWidth);
    vec3 CellCenterCoordPercent = (vec3(CellId) + vec3(0.5)) / vec3(PushConst.m_VoxelsPerClipMapWidth);
    vec3 CellCenterCoord = mix(vec3(PushConst.m_ClipMapRadius)*-1.0, vec3(PushConst.m_ClipMapRadius), CellCenterCoordPercent);
    vec3 CellExtent = vec3(1.0) * CellWidth;

    float CullingAcceptTh = 1.44 * CellWidth + kAyanami_ObjectGridCellQueryInterpolationRange * CellWidth;
    float CullingAcceptThSq = CullingAcceptTh * CullingAcceptTh;
    float CellCullingAcceptTh = kAyanami_ObjectGridCellQueryInterpolationRange * CellWidth;
    float CellCullingAcceptThSq = CellCullingAcceptTh * CellCullingAcceptTh;

    uint CellLoc = ifrit_ToCellId(CellId, uvec3(PushConst.m_VoxelsPerClipMapWidth));
    GetResource(BObjectCell, PushConst.m_CellDataId).m_Cell[CellLoc] = uvec4(0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF);
    barrier();
    
#if AYANAMI_OBJECT_GRID_CULL
    for(uint T=0; T<NumCullingPasses; T++){

        if(ifrit_IsFirstLane()){
            LocalSharedCullResultCount = 0;
        }
        barrier();

        // Cull mdfs to Tiles
        uint i;
        uint startId = T * kAyanami_ObjectGridCellMaxCullObjPerPass;
        uint endId = min(startId + kAyanami_ObjectGridCellMaxCullObjPerPass, PushConst.m_NumTotalMeshDF);
        for(i=startId+LocalId; i<endId; i+= LocalSize){
            MeshDFDesc MdfDesc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescListId).m_Data[i];
            MeshDFMeta MdfMeta = GetResource(BMeshDFMeta, MdfDesc.m_MdfMetaId).m_Data;
            mat4 localToWorld = GetResource(BLocalTransform, MdfDesc.m_TransformId).m_LocalToWorld;

            vec3 BoxLT = MdfMeta.bboxMin.xyz;
            vec3 BoxRB = MdfMeta.bboxMax.xyz;
            vec3 BoxCenterMS = (BoxLT + BoxRB) * 0.5;
            vec3 BoxExtentMS = (BoxRB - BoxLT);

            vec4 BoxCenterWSH = (localToWorld * vec4(BoxCenterMS, 1.0));
            vec3 BoxCenterWS = BoxCenterWSH.xyz / BoxCenterWSH.w;
            vec3 BoxExtentWS = BoxExtentMS * GetResource(BLocalTransform, MdfDesc.m_TransformId).m_MaxScale.xyz;

            float SqDist = ifrit_AabbSquaredDistance(TileCenterCoord, TileExtent, BoxCenterWS, BoxExtentWS);
            //if(SqDist < CullingAcceptThSq){
                uint LocalIndex = atomicAdd(LocalSharedCullResultCount, 1);
                if(LocalIndex < kAyanami_ObjectGridCellMaxCullObjPerPass){
                    LocalSharedCullResult[LocalIndex] = i;
                }
            //}
        }
        barrier();
        // compose mdfs to Cells
        for(i=0;i<LocalSharedCullResultCount;i++){
            uint MeshId = LocalSharedCullResult[i];
            MeshDFDesc MdfDesc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescListId).m_Data[MeshId];
            MeshDFMeta MdfMeta = GetResource(BMeshDFMeta, MdfDesc.m_MdfMetaId).m_Data;
            mat4 localToWorld = GetResource(BLocalTransform, MdfDesc.m_TransformId).m_LocalToWorld;
            mat4 WorldToLocal = GetResource(BLocalTransform, MdfDesc.m_TransformId).m_WorldToLocal;
            vec3 BoxLT = MdfMeta.bboxMin.xyz;
            vec3 BoxRB = MdfMeta.bboxMax.xyz;
            vec3 BoxCenterMS = (BoxLT + BoxRB) * 0.5;
            vec3 BoxExtentMS = (BoxRB - BoxLT);

            vec4 BoxCenterWSH = (localToWorld * vec4(BoxCenterMS, 1.0));
            vec3 BoxCenterWS = BoxCenterWSH.xyz / BoxCenterWSH.w;
            vec3 BoxExtentWS = BoxExtentMS * GetResource(BLocalTransform, MdfDesc.m_TransformId).m_MaxScale.xyz;

            float SqDist = ifrit_AabbSquaredDistance(CellCenterCoord, CellExtent, BoxCenterWS, BoxExtentWS);
            //if(SqDist < CellCullingAcceptThSq){
                //might be a candidate to this grid
                vec3 MeshMaxScale = vec3(GetResource(BLocalTransform, MdfDesc.m_TransformId).m_MaxScale.xyz);
                float HitDist = ClosestDistanceToSDF(MdfMeta, CellCenterCoord, MeshMaxScale,WorldToLocal);
                AddObjectToGridCell(MeshId, CellLoc, HitDist, PushConst.m_ClipMapRadius * 2.0);
                SortGridCell(CellLoc);
            //}
        }
        barrier();
    }
#else
    uint i;
    for(i=0;i<PushConst.m_NumTotalMeshDF;i++){
        uint MeshId = i;
        MeshDFDesc MdfDesc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescListId).m_Data[MeshId];
        MeshDFMeta MdfMeta = GetResource(BMeshDFMeta, MdfDesc.m_MdfMetaId).m_Data;
        mat4 localToWorld = GetResource(BLocalTransform, MdfDesc.m_TransformId).m_LocalToWorld;
        mat4 WorldToLocal = GetResource(BLocalTransform, MdfDesc.m_TransformId).m_WorldToLocal;
        vec3 BoxLT = MdfMeta.bboxMin.xyz;
        vec3 BoxRB = MdfMeta.bboxMax.xyz;
        vec3 BoxCenterMS = (BoxLT + BoxRB) * 0.5;
        vec3 BoxExtentMS = (BoxRB - BoxLT);

        vec4 BoxCenterWSH = (localToWorld * vec4(BoxCenterMS, 1.0));
        vec3 BoxCenterWS = BoxCenterWSH.xyz / BoxCenterWSH.w;
        vec3 BoxExtentWS = BoxExtentMS * GetResource(BLocalTransform, MdfDesc.m_TransformId).m_MaxScale.xyz;

        float SqDist = ifrit_AabbSquaredDistance(CellCenterCoord, CellExtent, BoxCenterWS, BoxExtentWS);
        if(SqDist < CellCullingAcceptThSq){
            //might be a candidate to this grid
            vec3 MeshMaxScale = vec3(GetResource(BLocalTransform, MdfDesc.m_TransformId).m_MaxScale.xyz);
            float HitDist = ClosestDistanceToSDF(MdfMeta, CellCenterCoord, MeshMaxScale,WorldToLocal);
            AddObjectToGridCell(MeshId, CellLoc, HitDist, PushConst.m_ClipMapRadius*0.5 );
            SortGridCell(CellLoc);
        }
    }

#endif
}