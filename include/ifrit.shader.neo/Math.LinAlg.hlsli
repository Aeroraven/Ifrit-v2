
#pragma once
#include "ifrit.shader.neo/Common.hlsli"

namespace IfritShader{
namespace Math{

    // Begin wrappers
    float3 Cross(float3 a, float3 b)
    {
        return cross(a, b);
    }

    float Cross(float a, float b)
    {
        return a * b;
    }

    float3 Mul(float3x3 a, float3 b)
    {
        return mul(a, b);
    }

    float Mul(float a, float b)
    {
        return a * b;
    }


    // End wrappers

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

    float Inverse(float m)
    {
        return rcp(m);
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

    // https://gist.github.com/mattatz/86fff4b32d198d0928d0fa4ff32cf6fa
    float4x4 Inverse(float4x4 m) {
        float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
        float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
        float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
        float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];
    
        float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
        float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
        float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
        float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;
    
        float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
        float idet = 1.0f / det;
    
        float4x4 ret;
    
        ret[0][0] = t11 * idet;
        ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
        ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
        ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;
    
        ret[1][0] = t12 * idet;
        ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
        ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
        ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;
    
        ret[2][0] = t13 * idet;
        ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
        ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
        ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;
    
        ret[3][0] = t14 * idet;
        ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
        ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
        ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;
    
        return ret;
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

    float4x4 Transpose(float4x4 m)
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

    float2x2 ClampDiag(float2x2 A, float minValue, float maxValue)
    {
        return float2x2(
            clamp(A._11, minValue, maxValue), A._12,
            A._21, clamp(A._22, minValue, maxValue)
        );
    }

    float3x3 ClampDiag(float3x3 A, float minValue, float maxValue)
    {
        return float3x3(
            clamp(A._11, minValue, maxValue), A._12, A._13,
            A._21, clamp(A._22, minValue, maxValue), A._23,
            A._31, A._32, clamp(A._33, minValue, maxValue)
        );
    }

    float DiagProductRelative(float3x3 A, float3x3 B)
    {
        return (A._11/B._11) * (A._22/B._22) * (A._33/B._33);
    }

    float DiagProductRelative(float2x2 A, float2x2 B)
    {
        return (A._11/B._11) * (A._22/B._22);
    }
}
}