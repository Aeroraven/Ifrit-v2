
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
#include "SamplerUtils.SharedConst.h"
#include "Math.SphericalHarmonics.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"


layout(push_constant) uniform UPushConstant{
    vec4 m_WorldBoundMin;
    vec4 m_WorldBoundMax;
    uint m_MeshDFDescListId;
    uint m_PerFrameId;
    uint m_TotalMdfCount;
    uint m_GridVpUAV;
    uint m_MaxMdfsPerGrid;
    uint m_NumGridsPerSlice;
    uint m_ScatterCounterUAV;
    uint m_ScatterOutputUAV;
    uint m_NumTilesWidth;
} PushConst;

RegisterStorage(BScatterOutput,{
    uint m_List[];
});

RegisterStorage(BScatterCounter,{
    uint m_Counter[];
}); 

layout(location = 0) in flat uint vInstanceId;
layout(location = 1) in flat uint vZSlice;

layout(location = 0) out float pDummy;

void main(){
    uvec2 TileXY = uvec2(gl_FragCoord.xy - 0.5);
    uint TileIdInSlice = TileXY.x + TileXY.y * PushConst.m_NumTilesWidth;
    uint TileId = TileIdInSlice + vZSlice * PushConst.m_NumGridsPerSlice;
    uint TileOffset = TileId * PushConst.m_MaxMdfsPerGrid;

    uint PropPos = atomicAdd(GetResource(BScatterCounter, PushConst.m_ScatterCounterUAV).m_Counter[TileId], 1u);
    if(PropPos >= PushConst.m_MaxMdfsPerGrid){
        return;
    }
    uint OverPos = PropPos + TileOffset;
    GetResource(BScatterOutput, PushConst.m_ScatterOutputUAV).m_List[OverPos] = vInstanceId;
    pDummy = 1.0;
}