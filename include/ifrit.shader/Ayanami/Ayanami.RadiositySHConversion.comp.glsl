
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
#include "Math.SphericalHarmonics.glsl"
#include "Math.Sampling.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(
    local_size_x = kAyanamiSphericalHarmonicsCvtKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec2 m_TraceCoordJitter;
    vec2 m_ProbeCenterJitter;
    uint m_CardAtlasResolution;
    uint m_CardResolution;
    uint m_NumTotalCards;
    uint m_CardDepthAtlasSRV;
    uint m_CardNormalAtlasSRV;
    uint m_AllCardObjDataId;
    uint m_AllMeshDFDataId;
    uint m_FilteredRadianceAtlasUAV;
    uint m_RWRadiosityProbeSHAtlasRUAV;
    uint m_RWRadiosityProbeSHAtlasGUAV;
    uint m_RWRadiosityProbeSHAtlasBUAV;
    uint m_TotalProbes;
}PushConst;

#include "Ayanami/Ayanami.Radiosity.Shared.glsl"

void WriteSHAtlas(uint ProbeIndex, MTwoBandSH_RGB SHCoefs){
    ivec2 WriteLocation = GetProbeSHAtlasCoord(ProbeIndex, PushConst.m_CardAtlasResolution, PushConst.m_CardResolution);
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_RWRadiosityProbeSHAtlasRUAV), ivec2(WriteLocation), SHCoefs.m_R.m_Coef);
    //imageStore(GetUAVImage2DRGBA32F(PushConst.m_RWRadiosityProbeSHAtlasRUAV), ivec2(WriteLocation), vec4(ProbeIndex));
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_RWRadiosityProbeSHAtlasGUAV), ivec2(WriteLocation), SHCoefs.m_G.m_Coef);
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_RWRadiosityProbeSHAtlasBUAV), ivec2(WriteLocation), SHCoefs.m_B.m_Coef);
}

void main(){
    uint GlobalId = gl_GlobalInvocationID.x; //Probe Id
    if(GlobalId >= PushConst.m_TotalProbes) return;
    uint ProbeRayStart = kAyanami_RadiosityTracesPerProbe * GlobalId;

    uint TileIndex;
    uvec2 OffsetInTile;
    uvec2 TraceRayCoord;
    AyaShared_RayTraceCoordToCardInfo(ProbeRayStart, PushConst.m_TraceCoordJitter, OffsetInTile, TileIndex, TraceRayCoord);

    RadiosityRayCardSample SampledData = AyaShared_GetRadiosityRayCardSample(TileIndex, OffsetInTile, PushConst.m_CardAtlasResolution,
        PushConst.m_CardResolution, PushConst.m_NumTotalCards, PushConst.m_CardDepthAtlasSRV,
        PushConst.m_CardNormalAtlasSRV, PushConst.m_AllCardObjDataId, PushConst.m_AllMeshDFDataId);
    
    MTwoBandSH_RGB SHCoefs = ifrit_ZeroSH2RGB();

    if(!SampledData.m_PresentInAtlas || !SampledData.m_ValidSample){
        WriteSHAtlas(GlobalId, SHCoefs);
        return;
    }

    uvec2 WriteSlot = AyaShared_GetRadianceSlot(TileIndex, OffsetInTile, TraceRayCoord, PushConst.m_CardAtlasResolution,
        PushConst.m_CardResolution);

    for(uint TraceX = 0;TraceX<kAyanami_RadiosityProbHemiRes;TraceX++){
        for(uint TraceY = 0;TraceY<kAyanami_RadiosityProbHemiRes;TraceY++){
            uvec2 TraceRayCoordS = uvec2(TraceX, TraceY);
            vec2 ProbeUV = (vec2(TraceRayCoordS) + vec2(0.5) + PushConst.m_ProbeCenterJitter) / float(kAyanami_RadiosityProbHemiRes);
            vec4 RayPDF = AyaShared_RadiosityGetRayPDF(ProbeUV);
            vec3 LocalRayDir = RayPDF.xyz;
            float PDF = RayPDF.w;
            mat3 TBN = ifrit_FrisvadONB(SampledData.m_WorldNormal);
            vec3 WorldRayDir = TBN * LocalRayDir;

            WriteSlot = AyaShared_GetRadianceSlot(TileIndex, OffsetInTile, TraceRayCoordS, PushConst.m_CardAtlasResolution,
                PushConst.m_CardResolution);
            vec3 FilteredRadiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_FilteredRadianceAtlasUAV), ivec2(WriteSlot)).rgb;

            SHCoefs = ifrit_AddSH2RGB(SHCoefs, ifrit_MulSH2RGBColor(ifrit_SHBasis2EncodeRGB(WorldRayDir), FilteredRadiance / PDF));
            //SHCoefs = ifrit_AddSH2RGB(SHCoefs, ifrit_MulSH2RGBColor(ifrit_SHBasis2EncodeRGB(WorldRayDir), FilteredRadiance));
        }
    }
    SHCoefs = ifrit_MulSH2RGB(SHCoefs, 1.0 / float(kAyanami_RadiosityProbHemiRes * kAyanami_RadiosityProbHemiRes));
    WriteSHAtlas(GlobalId, SHCoefs);
}