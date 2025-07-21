
#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.hlsli"

namespace IfritShader {
namespace Math {

    float4x4 AxisAngleRotation(float3 axis, float angle)
    {
        float c = cos(angle), s = sin(angle);
        float t = 1 - c, x = axis.x, y = axis.y, z = axis.z;
        float4x4 Ret;
        Ret[0][0] = t * x * x + c;
        Ret[0][1] = t * x * y + s * z;
        Ret[0][2] = t * x * z - s * y;
        Ret[0][3] = 0;
        Ret[1][0] = t * x * y - s * z;
        Ret[1][1] = t * y * y + c;
        Ret[1][2] = t * y * z + s * x;
        Ret[1][3] = 0;
        Ret[2][0] = t * x * z + s * y;
        Ret[2][1] = t * y * z - s * x;
        Ret[2][2] = t * z * z + c;
        Ret[2][3] = 0;
        Ret[3][0] = 0;
        Ret[3][1] = 0;
        Ret[3][2] = 0;
        Ret[3][3] = 1;
        Ret       = Transpose(Ret);
        return Ret;
    }

    float4x4 EulerAngleToRotMatrix(float3 Euler)
    {
        float4x4 Ret = AxisAngleRotation(float3(1.0f,0.0f,0.0f), Euler.x);
        Ret = mul(AxisAngleRotation(float3(0.0f,1.0f,0.0f), Euler.y), Ret);
        Ret = mul(AxisAngleRotation(float3(0.0f,0.0f,1.0f), Euler.z), Ret);
        return Ret;
    }

    float4x4 Scale3DTransform(float3 scale)
    {
        float4x4 ScaleMatrix = Identity4();
        ScaleMatrix[0][0] = scale.x;
        ScaleMatrix[1][1] = scale.y;
        ScaleMatrix[2][2] = scale.z;
        return ScaleMatrix;
    }

    float4x4 Translation3DTransform(float3 Translation)
    {
        float4x4 TranslationMatrix = Identity4();
        TranslationMatrix[0][3] = Translation.x;
        TranslationMatrix[1][3] = Translation.y;
        TranslationMatrix[2][3] = Translation.z;
        return TranslationMatrix;
    }

    float4x4 GetModelToWorldMatrix(float3 Position, float3 Rotation, float3 Scale)
    {
        float4x4 Model = Scale3DTransform(Scale);
        Model = mul(EulerAngleToRotMatrix(Rotation), Model);
        Model = mul(Translation3DTransform(Position), Model);
        return Model;
    }
    
}}