#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.Geometry.hlsli"
#include "ifrit.shader.neo/Artemis/Contact/Contact.Common.hlsli"
#include "ifrit.shader.neo/Artemis/Contact/SATContact.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{
    struct FRectContactResult
    {
        float2 ContactPointA1;
        float2 ContactPointA2;
        float2 ContactPointB1;
        float2 ContactPointB2;
        float2 Normal;
        bool Collided;
        bool HasSecondContact;
    };
    
    FRectContactResult RectToRectContact2D(FRectVertex RectA, float RotA, FRectVertex RectB, float RotB)
    {
        float2x2 RotMatA = Math::GetRotationMatrix(RotA);
        float2x2 RotMatB = Math::GetRotationMatrix(RotB);
        float2x2 InvRotMatA = Math::Inverse(RotMatA);
        float2x2 InvRotMatB = Math::Inverse(RotMatB);

        Quadrilateral2D QuadAInWS;
        Quadrilateral2D QuadBInWS;
        float2 UnitQuad[4] = {
            float2(-1.0f, -1.0f),
            float2(1.0f, -1.0f),
            float2(1.0f, 1.0f),
            float2(-1.0f, 1.0f)
        };

        IFSHADER_UNROLL
        for (int i = 0; i < 4; ++i)
        {
            float2 PointA = UnitQuad[i] * RectA.HalfSize;
            QuadAInWS.Points[i] = RectA.Center + mul(RotMatA, PointA);
            float2 PointB = UnitQuad[i] * RectB.HalfSize;
            QuadBInWS.Points[i] = RectB.Center + mul(RotMatB, PointB);
        }

        SATQuadContactManifold2D ContactManifold = QuadToQuadContactManifoldSAT2D(QuadAInWS, QuadBInWS);
        FRectContactResult Result;
        Result.Collided = ContactManifold.Collided;
        Result.Normal = ContactManifold.ContactNormal;
        Result.HasSecondContact = ContactManifold.NumIncidentPoints > 1;
        Result.ContactPointA1 = ContactManifold.ContactPointsOnA[0];
        Result.ContactPointA2 = ContactManifold.ContactPointsOnA[1];
        Result.ContactPointB1 = ContactManifold.ContactPointsOnB[0];
        Result.ContactPointB2 = ContactManifold.ContactPointsOnB[1];

        return Result;
    }

}}}
