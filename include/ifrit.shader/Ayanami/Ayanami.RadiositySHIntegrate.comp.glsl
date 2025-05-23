
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

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(
    local_size_x = 1, 
    local_size_y = kAyanamiRadiosityIntegrateKernelSizeY, 
    local_size_z = kAyanamiRadiosityIntegrateKernelSizeY, 
) in;

layout(push_constant) uniform UPushConstant{
    uint m_TotalCards;
    uint m_CardResolution;
    uint m_CardAtlasResolution;
    uint m_PerFrameCBV;
    uint m_MeshDFDescIdUAV;
    uint m_CardNormalAtlasSRV;

    uint m_RadiosityProbeSHAtlasRUAV;
    uint m_RadiosityProbeSHAtlasGUAV;
    uint m_RadiosityProbeSHAtlasBUAV;

    uint m_SurfaceIndirectLightingUAV;
}PushConst;

ivec2 GetProbeSHAtlasCoord(uint ProbeIndex){
    uint TilesPerAtlasWidth = PushConst.m_CardAtlasResolution / kAyanami_CardTileWidth;
    uint ProbesPerAtlasWidth = kAyanami_RadiosityProbesPerCardTileWidth * TilesPerAtlasWidth;
    uint ProbeX = ProbeIndex % ProbesPerAtlasWidth;
    uint ProbeY = ProbeIndex / ProbesPerAtlasWidth;
    return ivec2(ProbeX, ProbeY);
}

MTwoBandSH_RGB ReadSHAtlas(uint ProbeIndex){
    ivec2 WriteLocation = GetProbeSHAtlasCoord(ProbeIndex);
    MTwoBandSH_RGB SHCoefs;
    SHCoefs.m_R.m_Coef = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_RadiosityProbeSHAtlasRUAV), WriteLocation);
    SHCoefs.m_G.m_Coef = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_RadiosityProbeSHAtlasGUAV), WriteLocation);
    SHCoefs.m_B.m_Coef = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_RadiosityProbeSHAtlasBUAV), WriteLocation);
    return SHCoefs;
}


void main(){
    uvec3 GlobalInvo = uvec2(gl_GlobalInvocationID.xyz);
    uvec2 InCardTileOffset = GlobalInvo.yz;
    uint CardTileId = GlobalInvo.x;
    
    uint CardsPerRow = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
    uint CardTilesPerCard = PushConst.m_CardResolution / kAyanami_CardTileWidth;
    uint CardId = CardTileId / CardTilesPerCard;

    // Locate the probe id for this pixel
    uint CardTilesPerRow = PushConst.m_CardAtlasResolution / kAyanami_CardTileWidth;
    uint CardTileIdX = CardTileId % CardTilesPerRow;
    uint CardTileIdY = CardTileId / CardTilesPerRow;

    uint ProbeRangeWidth = kAyanami_CardTileWidth / kAyanami_RadiosityProbesPerCardTileWidth;
    uint InTileProbeIdX = InCardTileOffset.x / ProbeRangeWidth;
    uint InTileProbeIdY = InCardTileOffset.y / ProbeRangeWidth;
    uint ProbesPerCardTile = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityProbesPerCardTileWidth;

    uint InTileProbeId = InTileProbeIdX + InTileProbeIdY * kAyanami_RadiosityProbesPerCardTileWidth;
    uint CardTileProbeIdStart = CardTileId * ProbesPerCardTile;
    uint ProbeId = CardTileProbeIdStart + InTileProbeId;

    MTwoBandSH_RGB SHCoefs = ReadSHAtlas(ProbeId);

    // Get the normal from the surface cache and evaluate the diffuse transfer
    uint CardTilePosX = CardTileIdX * kAyanami_CardTileWidth + InCardTileOffset.x;
    uint CardTilePosY = CardTileIdY * kAyanami_CardTileWidth + InCardTileOffset.y;
    vec2 CardNormalRG = SampleTexture2DLoad(PushConst.m_CardNormalAtlasSRV, sNearestClamp, ivec2(CardTilePosX, CardTilePosY)).xy;
    vec2 NormalRG = CardNormalRG * 2.0 - 1.0;
    float NormalB = sqrt(1.0 - dot(NormalRG, NormalRG));
    vec3 NormalMap = vec3(NormalRG, NormalB);
    vec3 LocalNormal = normalize(NormalMap);

    mat4 LocalToWorld = AyaShared_GetLocalToWorld(PushConst.m_MeshDFDescIdUAV, CardId/6);
    vec3 WorldNormal = normalize((LocalToWorld * vec4(LocalNormal, 0.0)).xyz);

    MTwoBandSH DiffuseTransfer = ifrit_SHCosineLobe2Encode(WorldNormal);

    float DiffuseR = ifrit_DotSH2(SHCoefs.m_R, DiffuseTransfer);
    float DiffuseG = ifrit_DotSH2(SHCoefs.m_G, DiffuseTransfer);
    float DiffuseB = ifrit_DotSH2(SHCoefs.m_B, DiffuseTransfer);

    vec3 DiffuseVal = vec3(DiffuseR, DiffuseG, DiffuseB);

    // Write the result to the output buffer
    ivec2 WriteLocation = ivec2(CardTilePosX, CardTilePosY);
    vec4 WriteVal = vec4(DiffuseVal, 1.0);
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_SurfaceIndirectLightingUAV), WriteLocation, WriteVal);
}