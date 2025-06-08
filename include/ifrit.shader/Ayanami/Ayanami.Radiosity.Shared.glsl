
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


ivec2 GetProbeSHAtlasCoord(uint ProbeIndex,uint CardAtlasResolution, uint CardResolution){

#if INTERNAL_AYANAMI_RADIOSITY_TILE_SHCONV_DEBUG
    uint ProbesPerTile = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityProbesPerCardTileWidth;
    uint TileIndex = ProbeIndex / ProbesPerTile;
    uint ProbeRemainder = ProbeIndex % ProbesPerTile;

    uint TilesPerCardWidth = CardResolution / kAyanami_CardTileWidth;
    uint TilesPerCard = TilesPerCardWidth * TilesPerCardWidth;
    uint CardIndex = TileIndex / TilesPerCard;

    uint ReqStoreWidthPerTileWidth = kAyanami_RadiosityProbesPerCardTileWidth;
    uint ReqStoreWidthPerCardWidth = ReqStoreWidthPerTileWidth * TilesPerCardWidth;

    uint InTileProbeX = ProbeRemainder % ReqStoreWidthPerTileWidth;
    uint InTileProbeY = ProbeRemainder / ReqStoreWidthPerTileWidth;
    uvec2 InTileProbeOffset = uvec2(InTileProbeX, InTileProbeY);

    uint TileRemainder = TileIndex % TilesPerCard;
    uint TileX = TileRemainder % TilesPerCardWidth;
    uint TileY = TileRemainder / TilesPerCardWidth;
    uvec2 TileOffset = uvec2(TileX, TileY) * ReqStoreWidthPerTileWidth;

    uint CardPerAltasWidth = CardAtlasResolution / CardResolution;
    uint CardX = CardIndex % CardPerAltasWidth;
    uint CardY = CardIndex / CardPerAltasWidth;
    uvec2 CardOffset = uvec2(CardX, CardY) * ReqStoreWidthPerCardWidth;

    // Calculate the final coordinates in the SH atlas
    uvec2 FinalOffset = CardOffset + TileOffset + InTileProbeOffset;
    return ivec2(FinalOffset.x, FinalOffset.y);

#else

    uint TilesPerAtlasWidth = PushConst.m_CardAtlasResolution / kAyanami_CardTileWidth;
    uint ProbesPerAtlasWidth = kAyanami_RadiosityProbesPerCardTileWidth * TilesPerAtlasWidth;
    uint ProbeX = ProbeIndex % ProbesPerAtlasWidth;
    uint ProbeY = ProbeIndex / ProbesPerAtlasWidth;
    return ivec2(ProbeX, ProbeY);
#endif
}

vec4 AyaShared_RadiosityGetRayPDF(vec2 ProbeUV){
#if INTERNAL_AYANAMI_RADIOSITY_UNIFORM_DISTRIBUTION_DEBUG
    return ifrit_SampleUniformSphereWithPDF(ProbeUV);
#else
    return ifrit_SampleCosineHemisphereWithPDF(ProbeUV);
#endif
}