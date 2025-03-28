
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
#include "Ayanami/Ayanami.Shared.glsl"
#include "Ayanami/Ayanami.SharedConst.h"

layout(local_size_x = kAyanamiGlobalDFCompositeTileSize, local_size_y = kAyanamiGlobalDFCompositeTileSize, local_size_z = kAyanamiGlobalDFCompositeTileSize) in;

layout(push_constant) uniform UPushConstant{
    uvec4 m_GlobalDFTileRange;
    vec4 m_GlobalDFWorldRangeMin;
    vec4 m_GlobalDFWorldRangeMax;
    uint m_PerFrameDataId;
    uint m_GlobalDFVolumeId;
    uint m_NumMeshDF;
    uint m_MeshDFDescListId;
} PushConst;

RegisterUniform(BPerFrame,{
    PerFramePerViewData m_Data;
});

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

// Temporarily, global DF generation is done offline(or at startup) and stored in a 3D texture.
// So, virtual paging and streaming, including async loading/update, are not implemented yet.
void main(){
    uvec3 tID = gl_GlobalInvocationID;
    if(tID.x>=PushConst.m_GlobalDFTileRange.x || tID.y>=PushConst.m_GlobalDFTileRange.y || tID.z>=PushConst.m_GlobalDFTileRange.z) 
        return;
    
    vec3 CellNormalizedCoord = (vec3(tID)+0.5)/vec3(PushConst.m_GlobalDFTileRange);
    vec3 CellWorldCoord = mix(PushConst.m_GlobalDFWorldRangeMin.xyz, PushConst.m_GlobalDFWorldRangeMax.xyz, CellNormalizedCoord);

    uint TotalMeshes = PushConst.m_NumMeshDF;
    uint MeshDFDescListId = PushConst.m_MeshDFDescListId;
    uint GlobalDFVolumeId = PushConst.m_GlobalDFVolumeId;
    uint PerFrameDataId = PushConst.m_PerFrameDataId;

    float OptimalSDF = 1e10;


    for(uint CurMesh=0; CurMesh<TotalMeshes; CurMesh++){
        MeshDFDesc CurMeshDesc0 = GetResource(BMeshDFDesc, MeshDFDescListId).m_Data[CurMesh];
        mat4 WorldToLocal = GetResource(BLocalTransform, CurMeshDesc0.m_TransformId).m_WorldToLocal;
        vec3 MeshScale = GetResource(BLocalTransform, CurMeshDesc0.m_TransformId).m_MaxScale.xyz;
        float MeshMaxScale = max(max(MeshScale.x, MeshScale.y), MeshScale.z);

        uint MdfMetaId = CurMeshDesc0.m_MdfMetaId;
        MeshDFMeta MdfMeta = GetResource(BMeshDFMeta, MdfMetaId).m_Data;
        uint SDFId = MdfMeta.sdfId;

        vec4 CellLocalCoord = WorldToLocal*vec4(CellWorldCoord, 1.0);
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

        float SdfVal = texture(GetSampler3D(SDFId), ClampedUVW).r * MeshMaxScale;
        float TotalSdf = max(SdfVal + ToBoxAllPositive,ToBoxAll);

        OptimalSDF = min(OptimalSDF, TotalSdf);
    }

    imageStore(GetUAVImage3DR32F(PushConst.m_GlobalDFVolumeId), ivec3(tID), vec4(OptimalSDF, 0.0, 0.0, 0.0));
}