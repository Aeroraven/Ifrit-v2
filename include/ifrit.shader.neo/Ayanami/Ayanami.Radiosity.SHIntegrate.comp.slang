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

#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"
#include "ifrit.shader.neo/Math.SphericalHarmonics.hlsli"

namespace IfritShader{
namespace Ayanami{

    struct PushConstant_T
    {
        uint m_CardResolution;
        uint m_CardAtlasResolution;
        TRWStructuredBufferHandle<FMeshDFDescriptor> m_MeshDFDescIdUAV;
        TTexture2DHandle<float4> m_CardNormalAtlasSRV;
        TRWTexture2DHandle<float4> m_RadiosityProbeSHAtlasRUAV;
        TRWTexture2DHandle<float4> m_RadiosityProbeSHAtlasGUAV;
        TRWTexture2DHandle<float4> m_RadiosityProbeSHAtlasBUAV;
        TRWTexture2DHandle<float4> m_SurfaceIndirectLightingUAV;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    Math::FTwoBandSHRgb ReadSHAtlas(uint ProbeIndex){
        uint2 WriteLocation = GetRadiosityProbeSHAtlasCoord(ProbeIndex, PushConst.m_CardAtlasResolution, PushConst.m_CardResolution);
        Math::FTwoBandSHRgb SHCoefs;
        SHCoefs.m_Red.m_Coef = PushConst.m_RadiosityProbeSHAtlasBUAV.Load(WriteLocation);
        SHCoefs.m_Green.m_Coef = PushConst.m_RadiosityProbeSHAtlasGUAV.Load(WriteLocation);
        SHCoefs.m_Blue.m_Coef = PushConst.m_RadiosityProbeSHAtlasRUAV.Load(WriteLocation);
        return SHCoefs;
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        1,
        kAyanamiRadiosityIntegrateKernelSizeY,
        kAyanamiRadiosityIntegrateKernelSizeY
    )
    void RadiositySHIntegrateCS(uint3 ThreadID : SV_DispatchThreadID,
                                uint3 GroupID : SV_GroupID,
                                uint3 GroupThreadID : SV_GroupThreadID)
    {
        uint3 GlobalInvo = uint3(ThreadID.xyz);
        uint2 InCardTileOffset = GlobalInvo.yz;
        uint CardTileId = GlobalInvo.x;
        
        uint CardsPerRow = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
        uint CardTilesPerCardWidth = PushConst.m_CardResolution / kAyanami_CardTileWidth;
        uint CardTilesPerCard = CardTilesPerCardWidth * CardTilesPerCardWidth;
    
        uint CardId = CardTileId / CardTilesPerCard;
    
        uint CardX = CardId % CardsPerRow;
        uint CardY = CardId / CardsPerRow;
    
        uint TileRemainder = CardTileId % CardTilesPerCard;
        uint TileX = TileRemainder % CardTilesPerCardWidth;
        uint TileY = TileRemainder / CardTilesPerCardWidth;
    
        // Locate the probe id for this pixel
        uint CardTileIdX = TileX + CardX * CardTilesPerCardWidth;
        uint CardTileIdY = TileY + CardY * CardTilesPerCardWidth;
    
        uint ProbeRangeWidth = kAyanami_CardTileWidth / kAyanami_RadiosityProbesPerCardTileWidth;
        uint InTileProbeIdX = InCardTileOffset.x / ProbeRangeWidth;
        uint InTileProbeIdY = InCardTileOffset.y / ProbeRangeWidth;
        uint ProbesPerCardTile = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityProbesPerCardTileWidth;
    
        uint InTileProbeId = InTileProbeIdX + InTileProbeIdY * kAyanami_RadiosityProbesPerCardTileWidth;
        uint CardTileProbeIdStart = CardTileId * ProbesPerCardTile;
        uint ProbeId = CardTileProbeIdStart + InTileProbeId;
    
        Math::FTwoBandSHRgb SHCoefs = ReadSHAtlas(ProbeId);
    
        // Get the normal from the surface cache and evaluate the diffuse transfer
        uint CardTilePosX = CardTileIdX * kAyanami_CardTileWidth + InCardTileOffset.x;
        uint CardTilePosY = CardTileIdY * kAyanami_CardTileWidth + InCardTileOffset.y;
    
        float3 LocalNormal = PushConst.m_CardNormalAtlasSRV.Load(uint2(CardTilePosX, CardTilePosY),0).xyz;
        LocalNormal = LocalNormal * 2.0 - 1.0; 
    
        float4x4 LocalToWorld = PushConst.m_MeshDFDescIdUAV.Load(CardId/6).m_Transform.Load().m_LocalToWorld;
        float3 WorldNormal = normalize(mul(LocalToWorld, float4(LocalNormal, 0.0)).xyz);
    
        Math::FTwoBandSH DiffuseTransfer = Math::SHCosineLobe2Encode(WorldNormal);
    
        float DiffuseR = Math::DotSH2(SHCoefs.m_Red, DiffuseTransfer);
        float DiffuseG = Math::DotSH2(SHCoefs.m_Green, DiffuseTransfer);
        float DiffuseB = Math::DotSH2(SHCoefs.m_Blue, DiffuseTransfer);
    
        float3 DiffuseVal = float3(DiffuseR, DiffuseG, DiffuseB);
    
        // Write the result to the output buffer
        uint2 WriteLocation = uint2(CardTilePosX, CardTilePosY);
        float4 WriteVal = float4(DiffuseVal, 1.0);
        PushConst.m_SurfaceIndirectLightingUAV.Store(WriteLocation, WriteVal);
    }
}
}