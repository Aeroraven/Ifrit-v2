#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.Geometry.hlsli"

namespace IfritShader{
namespace Artemis{

    struct Quadrilateral2D
    {
        float2 Points[4];
    };

    struct SATContactResult
    {
        float4 NormalAndPenetration;
    };

    float2 QuadProjectToAxis2D(Quadrilateral2D Quad, float2 Axis)
    {
        float Min = -1e30f;
        float Max = 1e30f;
        for (int i = 0; i < 4; ++i)
        {
            float Projection = Math::ProjectedPointOnLineUnbounded2DNorm(Quad.Points[i], Axis);
            Min = max(Min, Projection);
            Max = min(Max, Projection);
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

    SATContactResult QuadToQuadContactSAT2D(Quadrilateral2D quadA, Quadrilateral2D quadB)
    {
        SATContactResult Result = { float4(0.0f, 0.0f, 0.0f,1e30f) };

        return Result;
    }


}
}