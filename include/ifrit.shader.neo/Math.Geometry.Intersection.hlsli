#pragma once

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.hlsli"

namespace IfritShader {
namespace Math {

    void Internal_CuboidPointIntersection(float3 Extent, float3 Point, out float3 ResultPt, out float3 ResultNormalX)
    {
        float3 DistToFaces = Extent - abs(Point);
        float3 Result = Point;
        float3 ResultNormal = float3(0.0f, 0.0f, 0.0f);
        float MinDist = min(min(DistToFaces.x, DistToFaces.y), DistToFaces.z);

        if(DistToFaces.x == MinDist)
        {
            ResultNormal.x = Point.x < 0.0f ? -1.0f : 1.0f;
            Result.x = Point.x < 0.0f ? -Extent.x : Extent.x;
        }
        else if(DistToFaces.y == MinDist)
        {
            ResultNormal.y = Point.y < 0.0f ? -1.0f : 1.0f;
            Result.y = Point.y < 0.0f ? -Extent.y : Extent.y;
        }
        else if(DistToFaces.z == MinDist)
        {
            ResultNormal.z = Point.z < 0.0f ? -1.0f : 1.0f;
            Result.z = Point.z < 0.0f ? -Extent.z : Extent.z;
        }
        ResultPt = Result;
        ResultNormalX = ResultNormal;

    }

    void Internal_RectanglePointIntersection(float2 Extent, float2 Point, out float2 ResultPt, out float2 ResultNormalX)
    {
        float2 DistToFaces = Extent - abs(Point);
        float2 Result = Point;
        float2 ResultNormal = float2(0.0f, 0.0f);
        float MinDist = min(DistToFaces.x, DistToFaces.y);

        if(DistToFaces.x == MinDist)
        {
            ResultNormal.x = Point.x < 0.0f ? -1.0f : 1.0f;
            Result.x = Point.x < 0.0f ? -Extent.x : Extent.x;
        }
        else if(DistToFaces.y == MinDist)
        {
            ResultNormal.y = Point.y < 0.0f ? -1.0f : 1.0f;
            Result.y = Point.y < 0.0f ? -Extent.y : Extent.y;
        }
        ResultPt = Result;
        ResultNormalX = ResultNormal;
    }


    bool CuboidPointCollisionInModelSpace3D(float3 Point, float3 Min, float3 Max, float3x3 Rotation, out float3 Intersection, out float3 Normal)
    {
        float3x3 InvRotation = Math::Inverse(Rotation);
        float3 CuboidCenter = (Min + Max) * 0.5f;   
        float3 WorldPointTranslated = Point - CuboidCenter;
        float3 LocalPoint = mul(InvRotation, WorldPointTranslated);

        float3 Extent = (Max - Min) * 0.5f;
        Intersection = float3(0.0f, 0.0f, 0.0f);
        Normal = float3(0.0f, 0.0f, 0.0f);
        
        if (abs(LocalPoint.x) <= Extent.x && abs(LocalPoint.y) <= Extent.y && abs(LocalPoint.z) <= Extent.z)
        {
            Internal_CuboidPointIntersection(Extent, LocalPoint, Intersection, Normal);
            return true; 
        }
        return false;
    }

    bool RectanglePointCollisionInModelSpace2D(float2 Point, float2 Min, float2 Max, float2x2 Rotation, out float2 Intersection, out float2 Normal)
    {
        float2x2 InvRotation = Math::Inverse(Rotation);
        float2 RectCenter = (Min + Max) * 0.5f;   
        float2 WorldPointTranslated = Point - RectCenter;
        float2 LocalPoint = mul(InvRotation, WorldPointTranslated);

        float2 Extent = (Max - Min) * 0.5f;
        Intersection = float2(0.0f, 0.0f);
        Normal = float2(0.0f, 0.0f);
        if (abs(LocalPoint.x) <= Extent.x && abs(LocalPoint.y) <= Extent.y)
        {
            Internal_RectanglePointIntersection(Extent, LocalPoint, Intersection, Normal);
            return true; 
        }
        return false;
    }

    bool SpherePointCollisionInModelSpace3D(float3 Point, float3 Center, float Radius, float3x3 Rotation, out float3 Intersection, out float3 Normal)
    {
        float3 WorldPointTranslated = Point - Center;
        float3x3 InvRotation = Math::Inverse(Rotation);
        float3 LocalPoint = mul(InvRotation, WorldPointTranslated);

        Intersection = float3(0.0f, 0.0f, 0.0f);
        Normal = float3(0.0f, 0.0f, 0.0f);

        float DistSquared = dot(LocalPoint, LocalPoint);
        if (DistSquared <= Radius * Radius)
        {
            float Dist = sqrt(DistSquared);
            Intersection = LocalPoint * (Radius / Dist);
            Normal = normalize(Intersection);
            return true; 
        }
        return false;
    }

    bool CirclePointCollisionInModelSpace2D(float2 Point, float2 Center, float Radius, float2x2 Rotation, out float2 Intersection, out float2 Normal)
    {
        float2 WorldPointTranslated = Point - Center;
        float2x2 InvRotation = Math::Inverse(Rotation);
        float2 LocalPoint = mul(InvRotation, WorldPointTranslated);

        Intersection = float2(0.0f, 0.0f);
        Normal = float2(0.0f, 0.0f);

        float DistSquared = dot(LocalPoint, LocalPoint);
        if (DistSquared <= Radius * Radius)
        {
            float Dist = sqrt(DistSquared);
            Intersection = LocalPoint * (Radius / Dist);
            Normal = normalize(Intersection);
            return true; 
        }
        return false;
    }


}}