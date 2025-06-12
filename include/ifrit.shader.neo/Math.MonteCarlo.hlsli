/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit.shader.neo/Common.hlsli"

namespace IfritShader{
namespace Math{
    
    float3x3 FrisvadOrthoNormalBasis(float3 Normal)
    {
        if(Normal.z<-0.999f)
        {
            float3 T1 = float3(0.0f, -1.0f, 0.0f);
            float3 B1 = float3(-1.0f, 0.0f, 0.0f);
            return float3x3(T1, B1, Normal);
        }
        else
        {
            float A = 1.0f * rcp(1.0f + Normal.z);
            float B = -Normal.x * Normal.y * A;
            float3 T1 = float3(1.0f - Normal.x * Normal.x * A, B, -Normal.x);
            float3 B1 = float3(B, 1.0f - Normal.y * Normal.y * A, -Normal.y);
            return float3x3(T1, B1, Normal);
        }
    }

    float4 SampleCosineHemisphereWithPDF(float2 UV)
    {
        float Phi = 2.0f * kPI * UV.x;
        float CosTheta = sqrt(UV.y);
        float SinTheta = sqrt(1.0f - CosTheta * CosTheta);
        float X = cos(Phi) * SinTheta;
        float Y = sin(Phi) * SinTheta;
        float Z = CosTheta;
        float PDF = CosTheta / kPI; // PDF
        return float4(X, Y, Z, PDF);
    }

    float4 SampleUniformSphereWithPDF(float2 UV)
    {
        float Phi = 2.0f * kPI * UV.x;
        float CosTheta = 1.0f - 2.0f * UV.y;
        float SinTheta = sqrt(1.0f - CosTheta * CosTheta);
        float X = cos(Phi) * SinTheta;
        float Y = sin(Phi) * SinTheta;
        float Z = CosTheta;
        float PDF = 1.0f / (4.0f * kPI); // PDF
        return float4(X, Y, Z, PDF);
    }

    float4 SampleCosineHemisphereWithPDF(float2 UV, float3 Normal)
    {
        float3 SampleH = SampleUniformSphereWithPDF(UV).xyz;
        float3 SampleH2 = normalize(Normal + SampleH);
        float PDF = dot(SampleH2, Normal) / kPI;
        return float4(SampleH2, PDF);
    }

    float3 ConcentricOctahedralTransform(float2 UV)
    {
        // https://zhuanlan.zhihu.com/p/408898601
        // https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
        // Port from ifrit.core.math

        // This implementation is based on shacklettbp/madrona
        // https://github.com/shacklettbp/madrona/blob/main/src/render/vk/shaders/utils.hlsl

        float2 U = 2.0f * UV - 1.0f;
        float D = 1.0f - abs(U.x) - abs(U.y);
        float R = 1.0f - abs(D);
        float R2 = R * R;
        float Phi = (R == 0.0f)?
            0.0f : (kPI * ((abs(U.y) - abs(U.x)) / R + 1.0f));
        float F = R * sqrt(2.0f - R2);
        float X = SignPreserveZero(U.x) * abs(F * cos(Phi));
        float Y = SignPreserveZero(U.y) * abs(F * sin(Phi));
        float Z = SignPreserveZero(D) * (1.0f - R2);
        return normalize(float3(X, Y, Z));
    }
}
}