#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.Geometry.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{
    struct FCircleContactResult
    {
        float2 ContactPoint1;
        float2 ContactPoint2;
        float2 Normal;
    };

    FCircleContactResult CircleToCircleContact2D(float2 CenterA, float RadiusA, float2 CenterB, float RadiusB)
    {
        FCircleContactResult Result = { float2(0.0f, 0.0f), float2(0.0f, 0.0f), float2(0.0f, 0.0f) };

        float2 Direction = CenterA - CenterB;
        float Distance = length(Direction);
        
        if (Distance < RadiusA + RadiusB)
        {
            Result.Normal = normalize(Direction);
            float PenetrationDepth = (RadiusA + RadiusB) - Distance;
            Result.ContactPoint1 = CenterA - Result.Normal * (RadiusA);
            Result.ContactPoint2 = CenterB + Result.Normal * (RadiusB);
        }
        
        return Result;
    }

}}}
