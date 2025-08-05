#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.Geometry.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.SpatialTransform.hlsli"
#include "ifrit.shader.neo/Artemis/Contact/Contact.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{
    struct FCircleRectContactResult
    {
        float2 ContactPoint1;
        float2 ContactPoint2;
        float2 Normal;
        bool Collided;
    };


    float2 GetShortestPointToRectOutside(float2 Point, float RectHalfW, float RectHalfH)
    {
        float PtX = clamp(Point.x, -RectHalfW, RectHalfW);
        float PtY = clamp(Point.y, -RectHalfH, RectHalfH);
        return float2(PtX, PtY);
    }
    float2 GetShortestPointToRectInside(float2 Point, float RectHalfW, float RectHalfH)
    {
        float2 AbsPoint = abs(Point);
        float2 DistToEdge = float2(RectHalfW, RectHalfH) - AbsPoint;
        float ShortestProp = min(DistToEdge.x, DistToEdge.y);
        if(ShortestProp == DistToEdge.x)
        {
            return float2(Point.x > 0 ? RectHalfW : -RectHalfW, Point.y);
        }
        else
        {
            return float2(Point.x, Point.y > 0 ? RectHalfH : -RectHalfH);
        }
    }


    FCircleRectContactResult CircleRectContact2D(float2 CenterA, float RadiusA, FRectVertex RectB, float RotDegB)
    {
        // Reference: Physics for Game Programmers : Robust Contact Creation for Physics Simulation
        // https://www.gdcvault.com/play/1022193/Physics-for-Game-Programmers-Robust
        FCircleRectContactResult Result;
        float2x2 RectBRotMat = Math::GetRotationMatrix(RotDegB);
        float2x2 RectBInvRotMat = Math::Inverse(RectBRotMat);
        float2 CenterAInBLS = mul(RectBInvRotMat, CenterA - RectB.Center);
        float HalfW = RectB.HalfSize.x;
        float HalfH = RectB.HalfSize.y;
        float2 ClosestPointOnRect = float2(0.0f);
        if (CenterAInBLS.x < -HalfW || CenterAInBLS.x > HalfW ||
            CenterAInBLS.y < -HalfH || CenterAInBLS.y > HalfH)
        {
            ClosestPointOnRect = GetShortestPointToRectOutside(CenterAInBLS, HalfW, HalfH);
        }
        else
        {
            ClosestPointOnRect = GetShortestPointToRectInside(CenterAInBLS, HalfW, HalfH);
        }
        float2 ContactVecLS = ClosestPointOnRect - CenterAInBLS;
        float2 ContactDirLS = normalize(ContactVecLS);
        Result.Collided  = length(ContactVecLS) < RadiusA;
    
        float2 ContactPoint1LS = CenterAInBLS + ContactDirLS * RadiusA;
        float2 ContactPoint2LS = ClosestPointOnRect;

        float2 ContactPoint1WS = mul(RectBRotMat, ContactPoint1LS) + RectB.Center;
        float2 ContactPoint2WS = mul(RectBRotMat, ContactPoint2LS) + RectB.Center;  

        Result.ContactPoint1 = ContactPoint1WS;
        Result.ContactPoint2 = ContactPoint2WS;
        Result.Normal = -normalize(ContactPoint2WS - ContactPoint1WS);
        return Result;
    }

    FCircleRectContactResult RectCircleContact2D(
        FRectVertex RectA, float RotDegA,
        float2 CenterB, float RadiusB
    )
    {
        FCircleRectContactResult Result = CircleRectContact2D(
            CenterB, RadiusB, RectA, RotDegA
        );

        FCircleRectContactResult NewResult;
        NewResult.ContactPoint1 = Result.ContactPoint2;
        NewResult.ContactPoint2 = Result.ContactPoint1;
        NewResult.Normal = -Result.Normal;
        NewResult.Collided = Result.Collided;
        return NewResult;
    }

}}}
