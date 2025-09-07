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

#pragma once
#include "Common.hlsli"

namespace IfritShader{
namespace Math{

    float ViewDepthToClipDepth(float ViewDepth, float CameraNear, float CameraFar)
    {
#ifndef IFSHADER_REVERSED_Z
        float ZDiff = rcp(CameraFar - CameraNear);
        float Dz = CameraFar * ViewDepth * ZDiff - CameraFar * CameraNear * ZDiff;
        return Dz;
#else
        #error "Reversed Z is not supported in this shader module."
#endif
    }

    float ClipDepthToViewDepth(float ClipDepth, float CameraNear, float CameraFar)
    {
#ifndef IFSHADER_REVERSED_Z
        float ZMul = CameraFar * CameraNear;
        float ZSub = CameraFar - CameraNear;
        return ZMul * rcp(CameraFar - ClipDepth*ZSub);
#else
        #error "Reversed Z is not supported in this shader module."
#endif
    }

    float SignedDistanceToPlane(float4 Plane, float3 Point)
    {
        return dot(Plane.xyz, Point) + Plane.w;
    }

    float PerspectiveLerp(float Z0, float Z1, float TVal)
    {
        float Numo = Z1 * Z0;
        float Deno = Z1 + TVal * (Z0 - Z1);
        return Numo * rcp(Deno);
    }

#define IFSHADER_PERSPECTIVE_LERP_DEF(T) \
    T PerspectiveLerp(T V0, T V1, float Z0, float Z1, float TVal) \
    { \
        float RcpZ0 = rcp(Z0); \
        float RcpZ1 = rcp(Z1);  \
        T Vz0 = V0 * RcpZ0; \
        T Vz1 = V1 * RcpZ1; \
        float Lp = lerp(RcpZ0, RcpZ1, TVal); \
        T Rp = lerp(Vz0, Vz1, TVal); \
        return Rp * rcp(Lp); \
    }

    IFSHADER_PERSPECTIVE_LERP_DEF(float)
    IFSHADER_PERSPECTIVE_LERP_DEF(float2)
    IFSHADER_PERSPECTIVE_LERP_DEF(float3)
    IFSHADER_PERSPECTIVE_LERP_DEF(float4)
#undef IFSHADER_PERSPECTIVE_LERP_DEF

    float AabbSquaredDistance(float3 Center1, float3 Extents1, float3 Center2, float3 Extents2)
    {
        float3 D = max(abs(Center1 - Center2) - (Extents1 + Extents2), 0.0f);
        return dot(D, D);
    }

    float4x4 CubeSpaceRemap(float3 SrcMin, float3 SrcMax, float3 DstMin, float3 DstMax)
    {
        float3 Scale = (DstMax - DstMin) / (SrcMax - SrcMin);
        float3 Offset = DstMin - SrcMin * Scale;
        
        float4 Col1 = float4(Scale.x, 0.0f, 0.0f, 0.0f);
        float4 Col2 = float4(0.0f, Scale.y, 0.0f, 0.0f);
        float4 Col3 = float4(0.0f, 0.0f, Scale.z, 0.0f);
        float4 Col4 = float4(Offset.x, Offset.y, Offset.z, 1.0f);
        return float4x4(Col1, Col2, Col3, Col4);
    }

}
}