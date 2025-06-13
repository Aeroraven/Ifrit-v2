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

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedConst.hlsli"
#include "ifrit.shader.neo/Math.RayTrace.hlsli"

namespace IfritShader {
namespace Ayanami {

    struct PushConstant_T
    {
        float4 m_GlobalDFBoxMin;
        float4 m_GlobalDFBoxMax;
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        TTexture3DHandle<float> m_GlobalDF;
        TRWTexture2DHandle<float4> m_OutputTex;
        uint m_RtH;
        uint m_RtW;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiGlobalDFRayMarchTileSize,
        kAyanamiGlobalDFRayMarchTileSize,
        1
    )
    void DebugGlobalDFRayMarchCS(uint3 ThreadID: SV_DispatchThreadID,
                                 uint3 GroupID: SV_GroupID,
                                 uint3 GroupThreadID: SV_GroupThreadID)
    {
        if (ThreadID.x >= PushConst.m_RtW || ThreadID.y >= PushConst.m_RtH)
            return;
        float2 UV = float2(ThreadID.x, ThreadID.y) / float2(PushConst.m_RtW, PushConst.m_RtH);

        // Calculate the ray origin and direction based on camera parameters
        PerFramePerViewData PerFrame = PushConst.m_PerFrame.Load();
        float Fov = PerFrame.m_CameraFovX;
        float Aspect = PerFrame.m_CameraAspect;
        float3 CameraPos = PerFrame.m_CameraPosition.xyz;
        float3 CameraFront = PerFrame.m_CameraFront.xyz;

        float3 ZAxis = normalize(CameraFront);
        float3 XAxis = normalize(cross(float3(0.0,1.0,0.0),ZAxis));
        float3 YAxis = normalize(cross(ZAxis,XAxis));

        float3x3 RotationMat = float3x3(XAxis, YAxis, ZAxis);

        if(ThreadID.x >= PushConst.m_RtW || ThreadID.y >= PushConst.m_RtH)
            return;

        float NdcX = -(2.0f * (float(ThreadID.x)+0.5f) / float(PushConst.m_RtW) - 1.0f) * Aspect;
        float NdcY = 1.0f - 2.0f * (float(ThreadID.y)+0.5f) / float(PushConst.m_RtH);
        float HalfFovTan = tan(Fov * 0.5f);

        float3 RayDir = normalize(float3(NdcX * HalfFovTan, NdcY * HalfFovTan, 1.0f));
        RayDir = mul(RotationMat, RayDir);

        float3 Normal = float3(0.0,0.0,0.0);
        float BestT = 1e20f;

        float3 RayOrigin = CameraPos;
        float3 Lb = PushConst.m_GlobalDFBoxMin.xyz;
        float3 Rt = PushConst.m_GlobalDFBoxMax.xyz;

        float HitT;
        float HitTMax;
        bool Hit = Math::RayBoxIntersection3D(RayOrigin,RayDir,Lb,Rt,HitT, HitTMax);

        //HitT = max(HitT, 0.0f);
        float3 NormalEps = float3(1.0f/256.0f, 1.0f/256.0f, 1.0f/256.0f) * 0.15f;
        bool Found = false;

        if(Hit)
        {
            for(int i=0; i<200 ;i++)
            {
                float3 Pt = RayOrigin + RayDir * HitT;
                float3 UVW = saturate((Pt - Lb) / (Rt - Lb));
                float Sdf = PushConst.m_GlobalDF.SampleLevel(ESamplerType::SLinearClamp,UVW,0.0f) - 0.005f;

                if(Sdf<0.005f)
                {
                    Found = true;
                    float Dx1 = PushConst.m_GlobalDF.SampleLevel(ESamplerType::SLinearClamp, UVW + float3(NormalEps.x, 0.0f, 0.0f),0.0f);
                    float Dx2 = PushConst.m_GlobalDF.SampleLevel(ESamplerType::SLinearClamp, UVW - float3(NormalEps.x, 0.0f, 0.0f),0.0f);
                    float Dy1 = PushConst.m_GlobalDF.SampleLevel(ESamplerType::SLinearClamp, UVW + float3(0.0f, NormalEps.y, 0.0f),0.0f);
                    float Dy2 = PushConst.m_GlobalDF.SampleLevel(ESamplerType::SLinearClamp, UVW - float3(0.0f, NormalEps.y, 0.0f),0.0f);
                    float Dz1 = PushConst.m_GlobalDF.SampleLevel(ESamplerType::SLinearClamp, UVW + float3(0.0f, 0.0f, NormalEps.z),0.0f);
                    float Dz2 = PushConst.m_GlobalDF.SampleLevel(ESamplerType::SLinearClamp, UVW - float3(0.0f, 0.0f, NormalEps.z),0.0f);

                    Normal.x = Dx1 - Dx2;
                    Normal.y = Dy1 - Dy2;
                    Normal.z = Dz1 - Dz2;

                    Normal = normalize(Normal);
                    break;
                }
                HitT+= max(1e-4f, Sdf * 0.3f);
                if(HitT > HitTMax)
                {
                    break;
                }
            }
            PushConst.m_OutputTex.Store(ThreadID.xy, float4(Normal,0.0f));
        }
        else
        {
            PushConst.m_OutputTex.Store(ThreadID.xy, float4(Normal,1.0f));
        }
       
    }
}
}