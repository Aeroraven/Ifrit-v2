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

    bool LineSphereIntersection(float3 SegStart, float3 SegEnd, 
        float3 SphereCenter, float SphereRadius,
        out float3 IntersectionPoint)
    {
        // Initialize default value
        IntersectionPoint = float3(0.0f, 0.0f, 0.0f);
        
        // Check if either endpoint is inside the sphere - important for PBD
        float distStartToCenter = length(SegStart - SphereCenter);
        float distEndToCenter = length(SegEnd - SphereCenter);
        
        if (distStartToCenter <= SphereRadius) {
            IntersectionPoint = SegStart;
            return true;
        }
        
        if (distEndToCenter <= SphereRadius) {
            IntersectionPoint = SegEnd;
            return true;
        }
        
        float3 D = SegEnd - SegStart;  
        float SegLength = length(D);
        
        // Handle degenerate segment
        if (SegLength < 1e-6f) {
            return false; // Already checked if point is inside above
        }
        
        // Normalize direction
        float3 Dir = D / SegLength;
        float3 M = SegStart - SphereCenter;

        // Standard ray-sphere intersection
        float A = 1.0f;  
        float B = 2.0f * dot(Dir, M);
        float C = dot(M, M) - SphereRadius * SphereRadius;
        
        float Discriminant = B * B - 4.0f * A * C;
        if (Discriminant < 0.0f) {
            return false;
        }
        
        // Get both intersection points
        float SqrtDisc = sqrt(Discriminant);
        float T1 = (-B - SqrtDisc) / (2.0f * A);
        float T2 = (-B + SqrtDisc) / (2.0f * A);
        
        // Check both T1 and T2 independently
        bool validT1 = (T1 >= 0.0f && T1 <= SegLength);
        bool validT2 = (T2 >= 0.0f && T2 <= SegLength);
        
        // Choose closest valid intersection point
        if (validT1) {
            IntersectionPoint = SegStart + T1 * Dir;
            return true;
        } else if (validT2) {
            IntersectionPoint = SegStart + T2 * Dir;
            return true;
        }
        
        return false;
    }
}
}