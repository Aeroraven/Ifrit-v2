
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


#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"
#include "Math.Sampling.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(
    local_size_x = kAyanamiRadiosityTraceKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec4 m_GlobalDFBoxMin;
    vec4 m_GlobalDFBoxMax;
    vec2 m_TraceCoordJitter;
    vec2 m_ProbeCenterJitter;
    uint m_TraceRadianceAtlasUAV;
    uint m_GlobalDFSRV;
    uint m_CardResolution;
    uint m_CardAtlasResolution;
    uint m_CardDepthAtlasSRV;
    uint m_CardNormalAtlasSRV;
    uint m_CardLightingAtlasSRV;
    uint m_AllCardObjDataId;
    uint m_AllMeshDFDataId;
    uint m_NumTotalCards;
    uint m_GlobalDFResolution;
    uint m_VoxelsPerWidth;
    uint m_ObjectGridUAV;
}PushConst;

float TraceGlobalDF(vec3 RayOrigin, vec3 RayDir){
    float hitTime = AyaShared_RayMarchGlobalDF(RayOrigin,RayDir,PushConst.m_GlobalDFSRV,PushConst.m_GlobalDFBoxMin.xyz,
        PushConst.m_GlobalDFBoxMax.xyz,0.015,0.03,200);

    return hitTime;
}

void main(){
    uint tID = gl_LocalInvocationID.x;
    uvec2 gID = gl_WorkGroupID.xy;

    uint TileIndex;
    uvec2 OffsetInTile;
    uvec2 TraceRayCoord;
    AyaShared_RayTraceCoordToCardInfo(tID, PushConst.m_TraceCoordJitter, OffsetInTile, TileIndex, TraceRayCoord);

    uvec2 WriteSlot = AyaShared_GetRadianceSlot(TileIndex, OffsetInTile, TraceRayCoord, PushConst.m_CardAtlasResolution);

    RadiosityRayCardSample SampledData = AyaShared_GetRadiosityRayCardSample(TileIndex, OffsetInTile, PushConst.m_CardAtlasResolution,
        PushConst.m_CardResolution, PushConst.m_NumTotalCards, PushConst.m_CardDepthAtlasSRV,
        PushConst.m_CardNormalAtlasSRV, PushConst.m_AllCardObjDataId, PushConst.m_AllMeshDFDataId);
    if(!SampledData.m_PresentInAtlas){
        return;
    }

    vec3 RadianceVal = vec3(0.0);
    float HitDistance = 1e30;

    if(SampledData.m_ValidSample){
        // Prepare for global df tracing
        vec2 ProbeUV = (vec2(TraceRayCoord) + PushConst.m_ProbeCenterJitter) / float(kAyanami_RadiosityProbHemiRes);
        vec4 RayPDF = ifrit_SampleCosineHemisphereWithPDF(ProbeUV);
        vec3 LocalRayDir = RayPDF.xyz;
        float PDF = RayPDF.w;
        mat3 TBN = ifrit_FrisvadONB(SampledData.m_WorldNormal);
        vec3 WorldRayDir = TBN * LocalRayDir;

        // Here, trace!
        vec3 RayOrigin = SampledData.m_WorldPos + WorldRayDir * 0.01;
        vec3 RayDir = WorldRayDir;
        float HitTime = TraceGlobalDF(RayOrigin,RayDir);
        bool IsHit = HitTime > 1e-3;

        vec3 FinalRadiance = vec3(0.0);
        if(IsHit){
            CardSample HitSample = AyaShared_EvaluateGlobalDFHit(RayOrigin,RayDir,HitTime,
                PushConst.m_GlobalDFSRV,PushConst.m_GlobalDFBoxMin.xyz,PushConst.m_GlobalDFBoxMax.xyz,
                PushConst.m_GlobalDFResolution,PushConst.m_VoxelsPerWidth,PushConst.m_ObjectGridUAV,
                PushConst.m_AllMeshDFDataId,PushConst.m_AllCardObjDataId,PushConst.m_CardDepthAtlasSRV,
                PushConst.m_CardLightingAtlasSRV, PushConst.m_CardResolution, PushConst.m_CardAtlasResolution,
                kAyanamiObjectGridTileSize);

            // TODO: final lighting is yet to be implemented.
            // This will be considered later
            FinalRadiance = HitSample.m_Albedo.xyz;
        }

        // Write to atlas
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_TraceRadianceAtlasUAV), ivec2(WriteSlot), vec4(FinalRadiance, 1.0));
    }
}