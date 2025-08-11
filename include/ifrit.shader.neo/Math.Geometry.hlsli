
#pragma once
#include "ifrit.shader.neo/Common.hlsli"

namespace IfritShader{
namespace Math{

    float3 CalculateTriangleNormal(float3 p0, float3 p1, float3 p2)
    {
        float3 edge1 = p1 - p0;
        float3 edge2 = p2 - p0;
        return normalize(cross(edge1, edge2));
    }

    float2 CalculateSegmentNormal(float2 p0, float2 p1)
    {
        float2 edge = p1 - p0;
        return normalize(float2(-edge.y, edge.x)); // Perpendicular vector
    }

    float ShortestSignedDistanceToPlane(float3 Point, float3 PlaneNormal, float3 PlanePoint)
    {
        return dot(Point - PlanePoint, PlaneNormal);
    }

    float ShortestUnsignedDistanceToPlane(float3 Point, float3 PlaneNormal, float3 PlanePoint)
    {
        return abs(ShortestSignedDistanceToPlane(Point, PlaneNormal, PlanePoint));
    }

    bool ProjectedPointInTriangle(float3 Point, float3 p0, float3 p1, float3 p2, out float3 BarycentricCoords)
    {
        float3 v0 = p1 - p0;
        float3 v1 = p2 - p0;
        float3 v2 = Point - p0;

        float d00 = dot(v0, v0);
        float d01 = dot(v0, v1);
        float d11 = dot(v1, v1);
        float d20 = dot(v2, v0);
        float d21 = dot(v2, v1);

        float denom = d00 * d11 - d01 * d01;
        BarycentricCoords = float3(0.0f, 0.0f, 0.0f);
        if (denom == 0.0f)
            return false; // Degenerate triangle

        BarycentricCoords.y = (d11 * d20 - d01 * d21) / denom;
        BarycentricCoords.z = (d00 * d21 - d01 * d20) / denom;
        BarycentricCoords.x = 1.0f - BarycentricCoords.y - BarycentricCoords.z;

        return (BarycentricCoords.x >= 0.0f && BarycentricCoords.y >= 0.0f && BarycentricCoords.z >= 0.0f);
    }

    float ShortestSignedDistanceToLine2D(float2 Point, float2 LinePoint, float2 LineDirection)
    {
        float2 LineToPoint = Point - LinePoint;
        float2 Perpendicular = float2(-LineDirection.y, LineDirection.x);
        float Distance = dot(LineToPoint, Perpendicular) / length(LineDirection);
        return Distance;
    }

    float ShortestUnsignedDistanceToLine2D(float2 Point, float2 LinePoint, float2 LineDirection)
    {
        float Distance = ShortestSignedDistanceToLine2D(Point, LinePoint, LineDirection);
        return abs(Distance);
    }

    bool ProjectedPointInSegment2D(float2 Point, float2 SegmentStart, float2 SegmentEnd, out float t)
    {
        float2 SegmentDirection = SegmentEnd - SegmentStart;
        float SegmentLengthSquared = dot(SegmentDirection, SegmentDirection);
        t = 0.0f;
        if (SegmentLengthSquared == 0.0f)
            return false; // Degenerate segment

        float tValue = dot(Point - SegmentStart, SegmentDirection) / SegmentLengthSquared;
        if (tValue < 0.0f || tValue > 1.0f)
            return false; // Outside segment bounds

        t = tValue;
        return true;
    }

    float ProjectedPointOnLineUnbounded2D(float2 Point, float2 LineDir)
    {
        float2 LineDirNormalized = normalize(LineDir);
        return dot(Point, LineDirNormalized);
    }
    float ProjectedPointOnLineUnbounded2DNorm(float2 Point, float2 LineDirNormalized)
    {
        return dot(Point, LineDirNormalized);
    }

    float ShortestUnsignedDistanceToLine3D(float3 Point, float3 LinePoint, float3 LineDirection)
    {
        float3 LineToPoint = Point - LinePoint;
        float3 Perpendicular = cross(LineDirection, LineToPoint);
        return length(Perpendicular) / length(LineDirection);
    }
}}
