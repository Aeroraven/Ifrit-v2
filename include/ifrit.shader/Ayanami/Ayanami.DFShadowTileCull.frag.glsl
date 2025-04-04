
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

layout(location = 0) in flat uint meshId;

layout(push_constant) uniform UPushConstant{
    mat4 m_VP;
    uint m_NumMeshDF;
    uint m_MeshDFDescListId;
    uint m_NumTilesWidth;
    uint m_TileAtomics;
    uint m_ScatterOutput;
} PushConst;

RegisterStorage(BTileAtomics,{
    uint m_Data[];
});

RegisterStorage(BTileScatter,{
    uint m_Data[];
});

layout(location = 0) out float outDummy;

//TODO: MSAA for better quality
void main(){
    uvec2 tileXY = uvec2(gl_FragCoord.xy - 0.5);
    uint tileId = tileXY.x + tileXY.y * PushConst.m_NumTilesWidth;
    uint inListPos = atomicAdd(GetResource(BTileAtomics, PushConst.m_TileAtomics).m_Data[tileId], 1u);
    uint overPos = inListPos + PushConst.m_NumMeshDF * tileId;
    GetResource(BTileScatter, PushConst.m_ScatterOutput).m_Data[overPos] = meshId;
    outDummy = 0.0;
}
