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

    float TrowbridgeReitzGGX(float NdotH, float Roughness)
    {
        float a = Roughness * Roughness;
        float a2 = a * a;
        float NdotH2 = NdotH * NdotH;
        float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
        return a2 / (kPI * denom * denom);
    }

    float SmithSchlickGGX(float NdotV, float NdotL, float Roughness)
    {
        float AlphaRemap = (Roughness + 1.0f) * (Roughness + 1.0f) * 0.25f;
        float K = AlphaRemap * 0.5f;
        float G1 = NdotV * rcp((NdotV * (1.0f - K)) + K);
        float G2 = NdotL * rcp((NdotL * (1.0f - K)) + K);
        return G1 * G2;
    }

    float3 FresnelSchlick(float CosTheta, float3 F0)
    {
        return F0 + (1.0f - F0) * pow(1.0f - CosTheta, 5.0f);
    }

    float3 FresnelSchlickMetallic(float3 F0, float3 Albedo, float Metallic, float HdotV)
    {
        float3 F0Mix = lerp(F0, Albedo, Metallic);
        float3 F = FresnelSchlick(HdotV, F0Mix);
        return F;
    }

    float3 CookTorranceBRDF(float3 F,float G,float D,float NdotV,float NdotL)
    {
        float3 Specular = F * (G * D) / (4.0f * NdotV * NdotL + 1e-6f);
        return Specular;
    }

    float3 PBRDirectLighting(float3 LightFront, float3 WorldPosition, float3 WorldNormal, float3 Albedo,
         float ShadowVisibility,float3 PosToEyeDirection)
    {
        float3 LightDir = -normalize(LightFront);
        float3 V = normalize(PosToEyeDirection);
        float3 H = normalize(LightDir + V);

        float NdotH = max(0.0f, dot(WorldNormal, H));
        float Roughness = 0.62f;

        float D = TrowbridgeReitzGGX(NdotH, Roughness);

        float NdotV = max(0.0f, dot(WorldNormal, V));
        float NdotL = max(0.0f, dot(WorldNormal, LightDir));
        float G = SmithSchlickGGX(NdotV, NdotL, Roughness);

        float3 F0 = float3(0.04f, 0.04f, 0.04f); // Default F0 for metals
        float Metallic = 0.03f;
        float HdotV = max(0.0f, dot(H, V));

        float3 F = FresnelSchlickMetallic(F0, Albedo, Metallic, HdotV);

        float3 Ks = F;
        float3 Kd = 1.0 - Ks;
        Kd *= 1.0 - Metallic;

        float3 SpecularBRDF = CookTorranceBRDF(F, G, D, NdotV, NdotL);
        float3 DiffuseBRDF = Kd * Albedo / kPI;

        float3 Lo = (DiffuseBRDF + SpecularBRDF) * ShadowVisibility * NdotL * 5.0f;
        return Lo;
    }
}