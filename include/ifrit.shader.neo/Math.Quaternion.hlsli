#pragma once

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.hlsli"

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

    float3x3 QuaternionToRotationMatrix(FQuaternion q)
    {
        float3x3 rotationMatrix;
        float x = q.m_Quaternion.x;
        float y = q.m_Quaternion.y;
        float z = q.m_Quaternion.z;
        float w = q.m_Quaternion.w;

        rotationMatrix[0] = float3(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y));
        rotationMatrix[1] = float3(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x));
        rotationMatrix[2] = float3(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y));

        return rotationMatrix;
    }

    FQuaternion RotationMatrixToQuaternion(float3x3 RotM)
    {
        float3x3 RotMT = Math::Transpose(RotM);
        float m00 = RotMT[0][0];
        float m01 = RotMT[0][1];
        float m02 = RotMT[0][2];
        float m10 = RotMT[1][0];
        float m11 = RotMT[1][1];
        float m12 = RotMT[1][2];
        float m20 = RotMT[2][0];
        float m21 = RotMT[2][1];
        float m22 = RotMT[2][2];
        
        float t;
        FQuaternion q;
        
        if (m22 < 0)
        {
            if (m00 > m11)
            {
                t = 1 + m00 - m11 - m22;
                q.m_Quaternion = float4(t, m01 + m10, m20 + m02, m12 - m21);
            }
            else
            {
                t = 1 - m00 + m11 - m22;
                q.m_Quaternion = float4(m01 + m10, t, m12 + m21, m20 - m02);
            }
        }
        else
        {
            if (m00 < -m11)
            {
                t = 1 - m00 - m11 + m22;
                q.m_Quaternion = float4(m20 + m02, m12 + m21, t, m01 - m10);
            }
            else
            {
                t = 1 + m00 + m11 + m22;
                q.m_Quaternion = float4(m12 - m21, m20 - m02, m01 - m10, t);
            }
        }
        
        q.m_Quaternion *= 0.5 / sqrt(t);
        return q;
    }
    
}}

