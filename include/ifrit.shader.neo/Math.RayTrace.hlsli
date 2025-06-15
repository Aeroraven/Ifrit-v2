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
    bool RayBoxIntersection3D(float3 Origin, float3 Direction, float3 BoxMin, float3 BoxMax, out float TStart, out float TEnd)
    {
        float3 InvDir = rcp(Direction);
        float T0 = (BoxMin.x - Origin.x) * InvDir.x;
        float T1 = (BoxMax.x - Origin.x) * InvDir.x;
        float T2 = (BoxMin.y - Origin.y) * InvDir.y;
        float T3 = (BoxMax.y - Origin.y) * InvDir.y;
        float T4 = (BoxMin.z - Origin.z) * InvDir.z;
        float T5 = (BoxMax.z - Origin.z) * InvDir.z;

        float TMin = max(max(min(T0, T1), min(T2, T3)), min(T4, T5));
        float TMax = min(min(max(T0, T1), max(T2, T3)), max(T4, T5));
        if (TMax < 0 || TMin > TMax)
        {
            TStart = 0.0f;
            TEnd = 0.0f;
            return false; // No intersection
        }
        TStart = TMin;
        TEnd = TMax;
        return true;  
    }

    bool RayBoxIntersection2D(float2 Origin, float2 Direction, float2 BoxMin, float2 BoxMax, out float TStart, out float TEnd)
    {
        float2 InvDir = rcp(Direction);
        float T0 = (BoxMin.x - Origin.x) * InvDir.x;
        float T1 = (BoxMax.x - Origin.x) * InvDir.x;
        float T2 = (BoxMin.y - Origin.y) * InvDir.y;
        float T3 = (BoxMax.y - Origin.y) * InvDir.y;

        float TMin = max(max(min(T0, T1), min(T2, T3)), 0.0f);
        float TMax = min(max(T0, T1), max(T2, T3));
        if (TMax < 0 || TMin > TMax)
        {   
            TStart = -1.0f;
            TEnd = -1.0f;
            return false; // No intersection
        }
        TStart = TMin;
        TEnd = TMax;
        return true;  
    }

}
}