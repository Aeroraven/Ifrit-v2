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
#include "ifrit.shader.neo/HierarchicalZ/HzbBase.hlsli"
#include "ifrit.shader.neo/Math.Transforms.hlsli"
#include "ifrit.shader.neo/Math.RayTrace.hlsli"


namespace IfritShader{
namespace SSGI{

    struct ScreenSpaceTraceResult
    {
        float2 HitUV;
        bool FinalHit;
    };

    ScreenSpaceTraceResult ScreenSpaceTraceImpl(float3 RayStartVS,float3 RayEndVS, 
        float2 RayStartUV, float2 RayEndUV, float2 ScreenExtent,
        HzbDataHandle HzbData, TConstantBufferHandle<PerFramePerViewData> PerFrameHandle,
        uint MaxTraceIteration
    )
    {
        PerFramePerViewData PerFrame = PerFrameHandle.Load();
        float ClipNear = PerFrame.m_CameraNear;
        float ClipFar = PerFrame.m_CameraFar;

        float2 DiffUV = RayEndUV - RayStartUV;
        float2 DiffPixels = DiffUV * ScreenExtent;

        int2 DiffPixelsInt = int2(DiffPixels.x, DiffPixels.y);
        float MaxStepsF = max(abs(DiffPixelsInt.x), abs(DiffPixelsInt.y));
        uint MaxSteps = (uint)MaxStepsF + 1;

        float MinimalStep = 1.0f / (float(MaxSteps));

        int CurMip = 0;
        bool FinalHit = false;
        float2 HitUV = float2(0.0f, 0.0f);
        float DepthDiffVS = 0.0f;

        int ProceedSignX = DiffPixels.x > 0 ? 1 : -1;
        int ProceedSignY = DiffPixels.y > 0 ? 1 : -1;

        int ProceedStepX = DiffPixels.x >= 0 ? 1 : 0;
        int ProceedStepY = DiffPixels.y >= 0 ? 1 : 0;

        float CurStepF = 0.0f;
        int MaxIters = (int)(MaxTraceIteration);
        int CurIters = 0;

        float ProceedTexelX = 0.0;
        float ProceedTexelY = 0.0;
        float ProceedRefZ = 0.0;
        float ProceedCurZ = 0.0;

        float2 RayStartPx = RayStartUV * ScreenExtent;
        float2 RayEndPx = RayEndUV * ScreenExtent;
        float2 CurPx = RayStartPx;
        int2 RayStartUVInt = int2(RayStartPx.x, RayStartPx.y);

        bool MainDirectionX = abs(DiffPixels.x) >= abs(DiffPixels.y);
        float2 NormSSDirection = normalize(DiffPixels);

        float LastCurZ = 0.0;
        float LastRefZ = 0.0;
        float LastIter  = 0.0;

        float2 InvExtent = rcp(ScreenExtent);

        while(CurMip >= 0 && CurIters < MaxIters){
            CurIters += 1;
            CurStepF = (MainDirectionX)?
                (CurPx.x - RayStartPx).x / DiffPixels.x :
                (CurPx.y - RayStartPx).y / DiffPixels.y;
            float T = CurStepF;
            float2 CurUV = CurPx * InvExtent;

            float ReferenceZ = HzbData.GetPixel(CurMip, uint2(CurPx));
            bool ValidZ = ReferenceZ > 0.0f && ReferenceZ < 1.0f;
            ReferenceZ = Math::ClipDepthToViewDepth(ReferenceZ, ClipNear, ClipFar);
            float CurZ = Math::PerspectiveLerp(RayStartVS.z, RayEndVS.z, T);
            int2 CurUVIntMip = int2(CurPx) >> CurMip;

            if(ValidZ)
            {
                LastCurZ = CurZ;
                LastRefZ = ReferenceZ;
                LastIter = float(CurIters);
            }

            ProceedRefZ = ReferenceZ;
            ProceedCurZ = RayStartVS.z;

            bool IsCollided = false;

            if(CurZ - ReferenceZ>=-3e-4f && (CurMip!=0 || ValidZ) && T>0.0f){
                IsCollided = true;
            }

            if(IsCollided && CurMip == 0){
                FinalHit = true;
                HitUV = lerp(RayStartUV, RayEndUV, T);
                DepthDiffVS = CurZ - ReferenceZ;
                break;
            }

            if(!IsCollided){
                int NextTexelX = ((CurUVIntMip.x + ProceedStepX)<<CurMip) - RayStartUVInt.x;
                int NextTexelY = ((CurUVIntMip.y + ProceedStepY)<<CurMip) - RayStartUVInt.y;
    
                float StepX = float(NextTexelX) / float(NormSSDirection.x);
                float StepY = float(NextTexelY) / float(NormSSDirection.y);
                
                float NextStep;
                if(abs(NormSSDirection.x)<1e-6){
                    NextStep = StepY;
                }else if(abs(NormSSDirection.y)<1e-6){
                    NextStep = StepX;
                }else{
                    NextStep = min(StepX, StepY);
                }
                CurPx = float2(RayStartUVInt.xy) + NextStep * NormSSDirection.xy + float2(ProceedSignX,ProceedSignY) * float2(0.0001);
                
            #ifndef IFSHADER_SSGI_HIZ_DISABLE
                CurMip = min(CurMip+1, 6);
            #endif
            }else{
            #ifndef IFSHADER_SSGI_HIZ_DISABLE
                CurMip-=1;
            #endif
            }
        }

        if(abs(DepthDiffVS) > 0.05f){
            FinalHit = false;
            ScreenSpaceTraceResult Result;
            Result.HitUV = float2(0.0f, 0.0f);
            Result.FinalHit = false;
            return Result;
        }
        
        ScreenSpaceTraceResult Result;
        Result.HitUV = HitUV;
        Result.FinalHit = FinalHit;
        return Result;
    }

    ScreenSpaceTraceResult ScreenSpaceTrace(float3 RayDirWS, float3 RayOriginWS,
        TConstantBufferHandle<PerFramePerViewData> PerFrameHandle,
        HzbDataHandle HzbData, uint MaxTraceIteration,float2 ScreenExtent)
    {
        PerFramePerViewData PerFrame = PerFrameHandle.Load();
        bool ValidSample = true;
        float4x4 WorldToClip = PerFrame.m_WorldToClip;
        float4x4 ClipToWorld = PerFrame.m_ClipToWorld;
        float4x4 WorldToView = PerFrame.m_WorldToView;
        float4x4 ViewToClip = PerFrame.m_ViewToClip;
        float ClipNear = PerFrame.m_CameraNear;

        float4 OriginVS = mul(WorldToView, float4(RayOriginWS, 1.0f));
        float4 ProceedVS = mul(WorldToView, float4(RayOriginWS + RayDirWS * 5.0f, 1.0f));
        float4 RayDirVS = ProceedVS - OriginVS;

        if(ProceedVS.z < ClipNear+1e-3){
            // find O+td intersection with near plane.
            float t = (ClipNear+1e-3 - OriginVS.z) / RayDirVS.z;
            if(abs(RayDirVS.z) < 1e-4){
               ValidSample = false;
            }
            ProceedVS = OriginVS + RayDirVS * t;
        }

        float4 OriginCS = mul(WorldToClip, float4(RayOriginWS, 1.0f));
        float4 ProceedCS = mul(WorldToClip, float4(RayOriginWS + RayDirWS * 5.0f, 1.0f));
        
        float2 OriginNDCxy = OriginCS.xy / OriginCS.w;
        float2 ProceedNDCxy = ProceedCS.xy / ProceedCS.w;
        float2 OriginUV = (OriginNDCxy + 1.0) * 0.5f;
        float2 ProceedUV = (ProceedNDCxy + 1.0) * 0.5f;

        float2 ProceedDirUV = ProceedUV - OriginUV;
        float2 RayNDCIntersection;
        bool RayNDCIntersectionValid = Math::RayBoxIntersection2D(
            OriginUV,ProceedDirUV,
            float2(0.0f, 0.0f), float2(1.0f, 1.0f),
            RayNDCIntersection.x, RayNDCIntersection.y
        );
        if(OriginUV.x < 0.0 || OriginUV.x > 1.0 || OriginUV.y < 0.0 || OriginUV.y > 1.0){
            ValidSample = false;
        }
        if(RayNDCIntersection.x>0.0 || RayNDCIntersection.y < 0.0){
            ValidSample = false;
        }
        
        float3 RayTraceStartVS = OriginVS.xyz;
        float3 RayTraceEndVS = ProceedVS.xyz;
        float2 RayTraceStartUV = OriginUV;
        float2 RayTraceEndUV = ProceedUV;

        float3 RayTraceDirNew = normalize(RayTraceEndVS - RayTraceStartVS);
        float3 RayTraceDirOld = normalize(ProceedVS.xyz - OriginVS.xyz);
        
        if(!ValidSample)
        {
            ScreenSpaceTraceResult Result;
            Result.HitUV = float2(0.0f, 0.0f);
            Result.FinalHit = false;
            return Result;
        }

        return ScreenSpaceTraceImpl(
            RayTraceStartVS, RayTraceEndVS, RayTraceStartUV, RayTraceEndUV,
            ScreenExtent,HzbData, PerFrameHandle, MaxTraceIteration
        );
    }
}
}
 