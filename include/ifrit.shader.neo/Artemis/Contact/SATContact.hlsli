#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.Geometry.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.SpatialTransform.hlsli"

namespace IfritShader{
namespace Artemis{

    struct Quadrilateral2D
    {
        float2 Points[4];
    };

    struct SATContactResult2D
    {
        float2 SeparatingAxis;
        float PenetrationDepth;
        bool Collided;
    };

    struct SATQuadContactManifold2D
    {
        float2 ContactPointsOnA[4];
        float2 ContactPointsOnB[4];
        float2 ContactNormal;
        int NumIncidentPoints;
        bool Collided;
    };

    float2 QuadProjectToAxis2D(Quadrilateral2D Quad, float2 Axis)
    {
        float Min = 1e30f;
        float Max = -1e30f;
        for (int i = 0; i < 4; ++i)
        {
            float Projection = Math::ProjectedPointOnLineUnbounded2DNorm(Quad.Points[i], Axis);
            Min = min(Min, Projection);
            Max = max(Max, Projection);
        }
        return float2(Min, Max);
    }
    float GetSATPenetrationLength2D(float2 RangeA, float2 RangeB)
    {
        float Overlap = min(RangeA.y, RangeB.y) - max(RangeA.x, RangeB.x);
        return (Overlap > 0.0f) ? Overlap : 0.0f;
    }
    float2 GetTestAxis2D(float2 PointA, float2 PointB)
    {
        float2 Edge = PointB - PointA;
        return normalize(float2(-Edge.y, Edge.x)); 
    }
    float2 GetQuadCenter2D(Quadrilateral2D Quad)
    {
        return (Quad.Points[0] + Quad.Points[1] + Quad.Points[2] + Quad.Points[3]) * 0.25f;
    }

    SATContactResult2D QuadToQuadContactSAT2D(Quadrilateral2D QuadA, Quadrilateral2D QuadB)
    {
        SATContactResult2D Result;
        Result.PenetrationDepth = 1e30f; 
        Result.SeparatingAxis = float2(0.0f, 0.0f);

        float2 CenterA = GetQuadCenter2D(QuadA);
        float2 CenterB = GetQuadCenter2D(QuadB);
        float2 CenterDiff = CenterB - CenterA;

        for (int i = 0; i < 4; ++i)
        {
            float2 AxisA = GetTestAxis2D(QuadA.Points[i], QuadA.Points[(i + 1) % 4]);
            float2 AxisB = GetTestAxis2D(QuadB.Points[i], QuadB.Points[(i + 1) % 4]);

            float2 RangeA = QuadProjectToAxis2D(QuadA, AxisA);
            float2 RangeB = QuadProjectToAxis2D(QuadB, AxisA);
            float PenetrationLength = GetSATPenetrationLength2D(RangeA, RangeB);
            if (PenetrationLength <= 0.0f)
            {
                Result.Collided = false;
                return Result; 
            }

            if (PenetrationLength < Result.PenetrationDepth)
            {
                Result.PenetrationDepth = PenetrationLength;
                Result.SeparatingAxis = AxisA;
            }

            RangeA = QuadProjectToAxis2D(QuadA, AxisB);
            RangeB = QuadProjectToAxis2D(QuadB, AxisB);
            PenetrationLength = GetSATPenetrationLength2D(RangeA, RangeB);
            if (PenetrationLength <= 0.0f)
            {
                Result.Collided = false;
                return Result;
            }

            if (PenetrationLength < Result.PenetrationDepth)
            {
                Result.PenetrationDepth = PenetrationLength;
                Result.SeparatingAxis = AxisB;
            }
        }   
        Result.Collided = Result.PenetrationDepth < 1e30f;

        if (Result.Collided)
        {
            Result.SeparatingAxis = normalize(Result.SeparatingAxis);
            if (dot(Result.SeparatingAxis, CenterDiff) < 0.0f)
            {
                Result.SeparatingAxis = -Result.SeparatingAxis; 
            }
        }
        return Result;
    }

    int GetReferenceFace(Quadrilateral2D Quad, float2 SATAxis, float2 PenetrationRegion, out float BestDot)
    {
        // reference face: argmin(dot(face,satAxis)), s.t. union(face,penetrationRegion) != empty
        float DotVal = 1e30f;
        int ReferenceFace = -1;
        for(int i=0;i<4;i++)
        {
            float2 FaceS = Quad.Points[i];
            float2 FaceE = Quad.Points[(i + 1) % 4];
            float2 FaceDir = normalize(FaceE - FaceS);
            float RefDot = abs(dot(FaceDir, SATAxis));
            if(RefDot < DotVal)
            {
                float2 FaceProjection = QuadProjectToAxis2D(Quad, SATAxis);
                if(FaceProjection.x < PenetrationRegion.y && FaceProjection.y > PenetrationRegion.x)
                {
                    DotVal = RefDot;
                    ReferenceFace = i;
                }
            }
        }
        BestDot = DotVal;
        return ReferenceFace;
    }

    bool IsPointInQuad2D(Quadrilateral2D Quad, float2 Point)
    {
        // time consuming, but there are just 4 points for a quad >_<
        float2 Edge1 = Quad.Points[1] - Quad.Points[0];
        float2 Edge2 = Quad.Points[2] - Quad.Points[1];
        float2 Edge3 = Quad.Points[3] - Quad.Points[2];
        float2 Edge4 = Quad.Points[0] - Quad.Points[3];

        float2 ToPoint1 = Point - Quad.Points[0];
        float2 ToPoint2 = Point - Quad.Points[1];
        float2 ToPoint3 = Point - Quad.Points[2];
        float2 ToPoint4 = Point - Quad.Points[3];

        return (Math::Cross2D(Edge1, ToPoint1) >= 0 &&
                Math::Cross2D(Edge2, ToPoint2) >= 0 &&
                Math::Cross2D(Edge3, ToPoint3) >= 0 &&
                Math::Cross2D(Edge4, ToPoint4) >= 0);
    }

    SATQuadContactManifold2D QuadToQuadContactManifoldSAT2DImpl(Quadrilateral2D Reference,Quadrilateral2D Incident,
        float2 SATAxis, int ReferenceFace)
    {
        SATQuadContactManifold2D Manifold;
        Manifold.Collided = true;
        Manifold.NumIncidentPoints = 0;
        Manifold.ContactNormal = SATAxis;
        for(int i=0;i<4;i++)
        {
            Manifold.ContactPointsOnA[i] = float2(0.0f, 0.0f);
            Manifold.ContactPointsOnB[i] = float2(0.0f, 0.0f);
        }
        for(int i=0;i<4;i++)
        {
            if(IsPointInQuad2D(Reference, Incident.Points[i]))
            {
                Manifold.ContactPointsOnB[Manifold.NumIncidentPoints] = Incident.Points[i];
                float2 FaceS = Reference.Points[ReferenceFace];
                float2 FaceE = Reference.Points[(ReferenceFace + 1) % 4];

                float T;
                bool InBound = Math::ProjectedPointInSegment2D(Incident.Points[i], FaceS, FaceE,T);
                if(InBound)
                {
                    Manifold.ContactPointsOnA[Manifold.NumIncidentPoints] =  T*(FaceE - FaceS) + FaceS;
                    Manifold.NumIncidentPoints++;
                }
            }
        }
        return Manifold;
    }

    SATQuadContactManifold2D QuadToQuadContactManifoldSAT2D(Quadrilateral2D QuadA, Quadrilateral2D QuadB)
    {
        SATContactResult2D ContactResult = QuadToQuadContactSAT2D(QuadA, QuadB);
        SATQuadContactManifold2D Manifold;
        Manifold.Collided = ContactResult.Collided;
        if(!ContactResult.Collided)
        {
            Manifold.NumIncidentPoints = 0;
            return Manifold;
        }
        // get the penetration region
        float2 ProjectionA = QuadProjectToAxis2D(QuadA, ContactResult.SeparatingAxis);
        float2 ProjectionB = QuadProjectToAxis2D(QuadB, ContactResult.SeparatingAxis);
        float2 PenetrationRegion = float2(
            max(ProjectionA.x, ProjectionB.x),
            min(ProjectionA.y, ProjectionB.y)
        );

        float BestDotA, BestDotB;
        int ReferenceFaceA = GetReferenceFace(QuadA, ContactResult.SeparatingAxis, PenetrationRegion, BestDotA);
        int ReferenceFaceB = GetReferenceFace(QuadB, ContactResult.SeparatingAxis, PenetrationRegion, BestDotB);

        // SATQuadContactManifold2D ManifoldA = QuadToQuadContactManifoldSAT2DImpl(QuadA, QuadB,
        //         ContactResult.SeparatingAxis, ReferenceFaceA);
        // Manifold.ContactNormal = ContactResult.SeparatingAxis;
        // Manifold.NumIncidentPoints = ManifoldA.NumIncidentPoints;
        // for(int i=0;i<Manifold.NumIncidentPoints;i++)
        // {
        //     Manifold.ContactPointsOnA[i] = ManifoldA.ContactPointsOnA[i];
        //     Manifold.ContactPointsOnB[i] = ManifoldA.ContactPointsOnB[i];
        // }

        if(BestDotA < BestDotB)
        {
            SATQuadContactManifold2D ManifoldA = QuadToQuadContactManifoldSAT2DImpl(QuadA, QuadB,
                 ContactResult.SeparatingAxis, ReferenceFaceA);
            Manifold.ContactNormal = ContactResult.SeparatingAxis;
            Manifold.NumIncidentPoints = ManifoldA.NumIncidentPoints;
            for(int i=0;i<Manifold.NumIncidentPoints;i++)
            {
                Manifold.ContactPointsOnA[i] = ManifoldA.ContactPointsOnA[i];
                Manifold.ContactPointsOnB[i] = ManifoldA.ContactPointsOnB[i];
            }
        }
        else
        {
            SATQuadContactManifold2D ManifoldB = QuadToQuadContactManifoldSAT2DImpl(QuadB, QuadA,
                 -ContactResult.SeparatingAxis, ReferenceFaceB);
            Manifold.ContactNormal = ContactResult.SeparatingAxis;
            Manifold.NumIncidentPoints = ManifoldB.NumIncidentPoints;
            for(int i=0;i<Manifold.NumIncidentPoints;i++)
            {
                Manifold.ContactPointsOnA[i] = ManifoldB.ContactPointsOnB[i];
                Manifold.ContactPointsOnB[i] = ManifoldB.ContactPointsOnA[i];
            }
        }
        return Manifold;
    }


}
}
