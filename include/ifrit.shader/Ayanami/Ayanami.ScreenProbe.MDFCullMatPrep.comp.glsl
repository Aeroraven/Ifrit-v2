
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
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"
#include "Math.SphericalHarmonics.glsl"
#include "Math.LinAlg.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

// The kernel is responsible for 2 things.
// 1. prepare the vp matrices for mesh df culling
// 2. prepare indirect draw args for the mesh df culling scattering pass

layout(
    local_size_x = kAyanamiScrProbeMDFCullPrepKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec4 m_WorldBoundMin;
    vec4 m_WorldBoundMax;
    uint m_IndirectDrawArgs;
    uint m_SlicesZ;
    uint m_GridVpUAV;
    uint m_TotalMdfCount;
}PushConst;

RegisterStorage(BGridVp,{
    mat4 m_VP[];
});

RegisterStorage(BIndirectDrawArgs,{
    uint m_VertexCount;
    uint m_InstanceCount;
    uint m_StartVertex;
    uint m_VertexOffset;
    uint m_StartInstance;
});

void main(){
    uint GlobalTid = gl_GlobalInvocationID.x;
    if(ifrit_IsGlobalFirstThread()){
        uint TotalInstanceToRender = PushConst.m_TotalMdfCount * PushConst.m_SlicesZ;
        GetResource(BIndirectDrawArgs, PushConst.m_IndirectDrawArgs).m_InstanceCount = TotalInstanceToRender;
        GetResource(BIndirectDrawArgs, PushConst.m_IndirectDrawArgs).m_VertexCount = 3u * 12u;
        GetResource(BIndirectDrawArgs, PushConst.m_IndirectDrawArgs).m_StartVertex = 0u;
        GetResource(BIndirectDrawArgs, PushConst.m_IndirectDrawArgs).m_StartInstance = 0u;
    }

    if(GlobalTid >= PushConst.m_SlicesZ){
        return;
    }

    float GridXMin = PushConst.m_WorldBoundMin.x;
    float GridYMin = PushConst.m_WorldBoundMin.y;
    float GridZMin = mix(PushConst.m_WorldBoundMin.z, PushConst.m_WorldBoundMax.z, float(GlobalTid) / float(PushConst.m_SlicesZ));
    float GridXMax = PushConst.m_WorldBoundMax.x;
    float GridYMax = PushConst.m_WorldBoundMax.y;
    float GridZMax = mix(PushConst.m_WorldBoundMin.z, PushConst.m_WorldBoundMax.z, float(GlobalTid + 1u) / float(PushConst.m_SlicesZ));

    // ortho map
    vec3 SrcMin = vec3(GridXMin, GridYMin, GridZMin);
    vec3 SrcMax = vec3(GridXMax, GridYMax, GridZMax);
    vec3 DstMin = vec3(-1.0, -1.0, -1.0);
    vec3 DstMax = vec3(1.0, 1.0, 1.0);

    mat4 OrthoVP = ifrit_CubeSpaceRemap(SrcMin, SrcMax, DstMin, DstMax);
    GetResource(BGridVp, PushConst.m_GridVpUAV).m_VP[GlobalTid] = OrthoVP;
}