
#pragma once
#include "ifrit.shader.neo/Common.hlsli"

namespace IfritShader{
namespace Math{

    float3x3 OuterProduct(float3 a, float3 b)
    {
        return float3x3(
            a.x * b.x, a.x * b.y, a.x * b.z,
            a.y * b.x, a.y * b.y, a.y * b.z,
            a.z * b.x, a.z * b.y, a.z * b.z
        );
    }

    float2x2 OuterProduct(float2 a, float2 b)
    {
        return float2x2(
            a.x * b.x, a.x * b.y,
            a.y * b.x, a.y * b.y
        );
    }

    float4x4 OuterProduct(float4 a, float4 b)
    {
        return float4x4(
            a.x * b.x, a.x * b.y, a.x * b.z, a.x * b.w,
            a.y * b.x, a.y * b.y, a.y * b.z, a.y * b.w,
            a.z * b.x, a.z * b.y, a.z * b.z, a.z * b.w,
            a.w * b.x, a.w * b.y, a.w * b.z, a.w * b.w
        );
    }
    
    float3x3 Identity3()
    {
        return float3x3(
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        );
    }

    float2x2 Identity2()
    {
        return float2x2(
            1.0f, 0.0f,
            0.0f, 1.0f
        );
    }

    float4x4 Identity4()
    {
        return float4x4(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        );
    }

    float Determinant(float2x2 m)
    {
        return m._11 * m._22 - m._12 * m._21;
    }

    float Determinant(float3x3 m)
    {
        return m._11 * (m._22 * m._33 - m._23 * m._32) -
               m._12 * (m._21 * m._33 - m._23 * m._31) +
               m._13 * (m._21 * m._32 - m._22 * m._31);
    }

    float3x3 Inverse(float3x3 m)
    {
        float det = Determinant(m);
        if (det == 0.0f)
            return Identity3(); // or handle singular matrix case appropriately

        float invDet = 1.0f / det;

        return float3x3(
            (m._22 * m._33 - m._23 * m._32) * invDet,
            (m._13 * m._32 - m._12 * m._33) * invDet,
            (m._12 * m._23 - m._13 * m._22) * invDet,
            (m._23 * m._31 - m._21 * m._33) * invDet,
            (m._11 * m._33 - m._13 * m._31) * invDet,
            (m._13 * m._21 - m._11 * m._23) * invDet,
            (m._21 * m._32 - m._22 * m._31) * invDet,
            (m._12 * m._31 - m._11 * m._32) * invDet,
            (m._11 * m._22 - m._12 * m._21) * invDet
        );
    }


    float2x2 Inverse(float2x2 m)
    {
        float det = Determinant(m);
        if (det == 0.0f)
            return Identity2(); // or handle singular matrix case appropriately

        float invDet = 1.0f / det;

        return float2x2(
            m._22 * invDet, -m._12 * invDet,
            -m._21 * invDet, m._11 * invDet
        );
    }

    float3x3 Transpose(float3x3 m)
    {
        return transpose(m);
    }

    float2x2 Transpose(float2x2 m)
    {
        return transpose(m);
    }

    bool HasNaN(float3x3 m)
    {
        return isnan(m._11) || isnan(m._12) || isnan(m._13) ||
               isnan(m._21) || isnan(m._22) || isnan(m._23) ||
               isnan(m._31) || isnan(m._32) || isnan(m._33);
    }
    bool HasNaN(float2x2 m)
    {
        return isnan(m._11) || isnan(m._12) ||
               isnan(m._21) || isnan(m._22);
    }
    
    float Trace(float3x3 m)
    {
        return m._11 + m._22 + m._33;
    }

    float Trace(float2x2 m)
    {
        return m._11 + m._22;
    }

    float Trace(float4x4 m)
    {
        return m._11 + m._22 + m._33 + m._44;
    }

    void QRDecomposition(float3x3 A, out float3x3 Q, out float3x3 R)
    {
        float3 a1 = (float3(A._11, A._21, A._31));
        float3 a2 = (float3(A._12, A._22, A._32));
        float3 a3 = (float3(A._13, A._23, A._33));

        float3 b1 = a1;
        float k21 = dot(a2,b1) / dot(b1, b1);

        float3 b2 = a2 - k21 * b1;
        float k31 = dot(a3,b1) / dot(b1, b1);
        float k32 = dot(a3,b2) / dot(b2, b2);
        
        float3 b3 = a3 - k31 * b1 - k32 * b2;

        float3 q1 = normalize(b1);
        float3 q2 = normalize(b2);
        float3 q3 = normalize(b3);

        Q = float3x3(q1, q2, q3);
        R = float3x3(
            length(b1), k21 * length(b1), k31 * length(b1),
            0.0f, length(b2), k32 * length(b2),
            0.0f, 0.0f, length(b3)
        );
    }

    void QRDecomposition(float2x2 A, out float2x2 Q, out float2x2 R)
    {
        float2 a1 = (float2(A._11, A._21));
        float2 a2 = (float2(A._12, A._22));

        float2 b1 = a1;
        float k21 = dot(a2,b1) / dot(b1, b1);

        float2 b2 = a2 - k21 * b1;

        float2 q1 = normalize(b1);
        float2 q2 = normalize(b2);

        Q = float2x2(q1, q2);
        R = float2x2(
            length(b1), k21 * length(b1),
            0.0f, length(b2)
        );
    }

}
}