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
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"

namespace IfritShader{
namespace Lighting{
    // Cascaded Shadow Mapping
    struct CascadedShadowMappingData
    {
        TConstantBufferHandle<PerFramePerViewData> m_Views[4];
        TTexture2DHandle<float> m_ShadowMap[4];
        uint4 m_ViewMapping; //Useless for shader, should be optimized
        float4 m_CsmStart;
        float4 m_CsmEnd;
        uint m_CsmNumSplits;
    };

    uint GetCSMSplitIndex(
        float3 ViewPosition,
        uint LightId,
        TRWStructuredBufferHandle<CascadedShadowMappingData> LightCollectionData
    )
    {
        CascadedShadowMappingData CsmData = LightCollectionData.Load(LightId);
        float Depth = ViewPosition.z;
        uint SplitId = 0;
        uint SplitIdNext = 0;
        for(int i = 0; i < CsmData.m_CsmNumSplits; i++)
        {
            if(Depth >= CsmData.m_CsmStart[i] && Depth < CsmData.m_CsmEnd[i])
            {
                SplitId = i;
                if(i < CsmData.m_CsmNumSplits - 1 && Depth > CsmData.m_CsmStart[i + 1])
                {
                    SplitId |= 0x80000000; // Mark as next split
                }
                break;
            }
        }
        return SplitId;
    }

    float2 GetCSMShadowVisibilityImpl(
        uint LightId,
        float3 WorldPosition,
        uint CsmSplitId,
        TRWStructuredBufferHandle<CascadedShadowMappingData> LightCollectionData
    )
    {
        CascadedShadowMappingData CsmData = LightCollectionData.Load(LightId);
        PerFramePerViewData LightView = CsmData.m_Views[CsmSplitId].Load();
        float4x4 LightVP = LightView.m_WorldToClip;

        float4 LightSpacePosition = mul(LightVP, float4(WorldPosition, 1.0f));
        float3 LightSpaceNDC = LightSpacePosition.xyz / LightSpacePosition.w;
        float2 LightSpaceUV = (LightSpaceNDC.xy * 0.5f) + 0.5f; 

        float ShadowMapZ = CsmData.m_ShadowMap[CsmSplitId].SampleLevel(ESamplerType::SNearestClamp, LightSpaceUV,0);
        float ReferenceZ = LightSpaceNDC.z;

        if(LightSpaceUV.x < 0.0f || LightSpaceUV.x > 1.0f ||
           LightSpaceUV.y < 0.0f || LightSpaceUV.y > 1.0f)
        {
            return float2(0.0f, 0.0f); // Outside shadow map bounds
        }

        // PCF here!
        float TotalVisibleSamples = 0.0f;
        float TotalShadowSamples = 0.0f;
        for(int dx = -1; dx <= 1; dx++)
        {
            for(int dy = -1; dy <= 1; dy++)
            {
                float2 SampleUV = LightSpaceUV + float2(dx, dy) * 0.5f / float(2048);
                if(SampleUV.x < 0.0f || SampleUV.x > 1.0f ||
                   SampleUV.y < 0.0f || SampleUV.y > 1.0f)
                {
                    continue; // Skip samples outside shadow map bounds
                }
                float ShadowMapZSample = CsmData.m_ShadowMap[CsmSplitId].SampleLevel(ESamplerType::SNearestClamp, SampleUV, 0);
            
            #ifndef IFSHADER_REVERSED_Z
                TotalVisibleSamples += (ReferenceZ < ShadowMapZSample + 1e-5f) ? 1.0f : 0.0f;
            #else
                TotalVisibleSamples += (ReferenceZ > ShadowMapZSample - 1e-5f) ? 1.0f : 0.0f;
            #endif
                TotalShadowSamples += 1.0f;

            }
        }
        return float2(TotalVisibleSamples / TotalShadowSamples, 1.0f);
    }

    float2 GetCSMShadowVisibility(
        float3 WorldPosition,
        float3 ViewPosition,
        uint LightId,
        TRWStructuredBufferHandle<CascadedShadowMappingData> LightCollectionData
    )
    {
        float2 AvgShadowVisibility = float2(0.0f, 0.0f);
        uint SplitId = GetCSMSplitIndex(ViewPosition, LightId, LightCollectionData);
        uint RequireNextLevel = SplitId & 0x80000000;
        SplitId &= 0x7FFFFFFF; // Clear the next level flag
        float CurCSMEnd = LightCollectionData.Load(LightId).m_CsmEnd[SplitId];
        float Fade = 0.0f;
        if(RequireNextLevel != 0)
        {
            float NextCSMStart = LightCollectionData.Load(LightId).m_CsmStart[SplitId + 1];
            Fade = 1.0 - (ViewPosition.z - CurCSMEnd) / (NextCSMStart - CurCSMEnd);
        }
        float DitherRand = Random::Simplex2D(WorldPosition.xy);
        if(DitherRand < Fade)
        {
            SplitId++;
        }
        AvgShadowVisibility = GetCSMShadowVisibilityImpl(
            LightId,
            WorldPosition,
            SplitId,
            LightCollectionData
        );
        return AvgShadowVisibility;

    }
}
}