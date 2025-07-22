#pragma once

#include "ifrit.shader.neo/Common.hlsli"

namespace IfritShader {
namespace Math{

    struct FQuaternion
    {
        float4 m_Quaternion;
    };

    FQuaternion IdentityQ()
    {
        FQuaternion q;
        q.m_Quaternion = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return q;
    }
    
    FQuaternion Mul(FQuaternion a, FQuaternion b)
    {
        FQuaternion result;
        result.m_Quaternion = float4(
            a.m_Quaternion.x * b.m_Quaternion.w + a.m_Quaternion.y * b.m_Quaternion.z - a.m_Quaternion.z * b.m_Quaternion.y + a.m_Quaternion.w * b.m_Quaternion.x,
            -a.m_Quaternion.x * b.m_Quaternion.z + a.m_Quaternion.y * b.m_Quaternion.w + a.m_Quaternion.z * b.m_Quaternion.x + a.m_Quaternion.w * b.m_Quaternion.y,
            a.m_Quaternion.x * b.m_Quaternion.y - a.m_Quaternion.y * b.m_Quaternion.x + a.m_Quaternion.z * b.m_Quaternion.w + a.m_Quaternion.w * b.m_Quaternion.z,
            -a.m_Quaternion.x * b.m_Quaternion.x - a.m_Quaternion.y * b.m_Quaternion.y - a.m_Quaternion.z * b.m_Quaternion.z + a.m_Quaternion.w * b.m_Quaternion.w
        );
        return result;
    }

    FQuaternion Conjugate(FQuaternion q)
    {
        FQuaternion result;
        result.m_Quaternion = float4(-q.m_Quaternion.xyz, q.m_Quaternion.w);
        return result;
    }

    float Length(FQuaternion q)
    {
        return length(q.m_Quaternion);
    }

    FQuaternion Normalize(FQuaternion q)
    {
        FQuaternion result;
        result.m_Quaternion = normalize(q.m_Quaternion);
        return result;
    }

    FQuaternion ToQuaternion(float4 quaternion)
    {
        FQuaternion result;
        result.m_Quaternion = quaternion;
        return result;
    }

    FQuaternion Add(FQuaternion a, FQuaternion b)
    {
        FQuaternion result;
        result.m_Quaternion = a.m_Quaternion + b.m_Quaternion;
        return result;
    }

    FQuaternion Mul(float scalar, FQuaternion q)
    {
        FQuaternion result;
        result.m_Quaternion = scalar * q.m_Quaternion;
        return result;
    }
}}