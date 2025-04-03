
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

layout(
    local_size_x = kAyanamiRadianceInjectionCardSizePerBlock, 
    local_size_y = kAyanamiRadianceInjectionCardSizePerBlock, 
    local_size_z = kAyanamiRadianceInjectionObjectsPerBlock 
    ) in;

layout(push_constant)  uniform PushConstData{
    uint totalCards;
    uint cardResolution;
    uint packedShadowMarkBits;
    uint totalLights;
    uint cardAtlasResolution;

    uint lightDataId;
    uint radianceOutId;
    uint cardDataId;
    uint depthAtlasSRVId;
} PushConst;

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});

void main(){
    // uvec3 tID = gl_GlobalInvocationID;
    // uint maxCardsInLine = PushConst.cardAtlasResolution / PushConst.cardResolution;
    // uint cardIndex_X = tID.z % maxCardsInLine;
    // uint cardIndex_Y = tID.z / maxCardsInLine;
    // uvec2 cardOffset = uvec2(cardIndex_X * PushConst.cardResolution, cardIndex_Y * PushConst.cardResolution);
    // uvec2 tileOffset = uvec2(tID.x, tID.y);
    // uvec2 overallOffset = cardOffset + tileOffset;

    // uint cardIndex = tID.z;
    // uint tileIndex = tID.x + tID.y * gl_WorkGroupSize.x;


    // mat4 atlasToLocal = GetResource(BAllCardData, PushConst.cardDataId).m_Mats[cardIndex].m_VPInv;
    
}