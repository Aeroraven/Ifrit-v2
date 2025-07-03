
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

    float DiagProduct(float3x3 A,float eps)
    {
        float d1 = max(A._11, eps);
        float d2 = max(A._22, eps);
        float d3 = max(A._33, eps);
        return d1 * d2 * d3;
    }

    float DiagProduct(float2x2 A, float eps)
    {
        float d1 = max(A._11, eps);
        float d2 = max(A._22, eps);
        return d1 * d2;
    }

}
}