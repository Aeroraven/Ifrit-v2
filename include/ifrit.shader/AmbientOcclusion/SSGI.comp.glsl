
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


#version 450

#include "Base.glsl"
#include "Bindless.glsl"
#include "AmbientOcclusion/AmbientOcclusion.Shared.h"
#include "Math.SphericalHarmonics.glsl"
#include "Math.Sampling.glsl"
#include "Math.RayUtils.glsl"
#include "SamplerUtils.SharedConst.h"

#include "Random/Random.WNoise2D.glsl"
#include "Random/Random.BlueNoise2D.glsl"

RegisterUniform(BPerframe,{
    PerFramePerViewData m_Data;
});

RegisterStorage(BHiZStorage,{
    uint m_Pad;
    uint m_Mip[];
});

layout(local_size_x = cSSGIThreadGroupSizeX, local_size_y = cSSGIThreadGroupSizeY, local_size_z = 1) in;

const bool kHizProceed = true; 
const bool kAllowSkylightFallback = true;
const uint kMaxTraceIters = 60; 
const float kRayProceedMax = 20.0;

layout(push_constant) uniform PushConstantSSGI{
    uint m_PerFrameCBV;
    uint m_NormalTexSRV; //SRV
    uint m_HizMinRefUAV; //Ref->UAVs
    uint m_HizMaxRefUAV; //Ref->UAVs
    uint m_AOTexUAV; //UAV
    uint m_FinalLightingSRV; //SRV, Last frame's final lighting
    uint m_HizTexW;
    uint m_HizTexH;
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_MaxMips;
    uint m_BlueNoiseSRV;
    uint m_AlbedoTexSRV;
} PushConst;


float GetHizDepth(ivec2 UV, uint Mip, bool Ranged){
    ivec2 MipUV = UV >> Mip;
    uint MipId = GetResource(BHiZStorage,PushConst.m_HizMinRefUAV).m_Mip[Mip];
    if(!Ranged){
        return imageLoad(GetUAVImage2DR32F(MipId), MipUV).r;
    }
    float t0 = imageLoad(GetUAVImage2DR32F(MipId), MipUV).r;
    float t1 = imageLoad(GetUAVImage2DR32F(MipId), MipUV+ivec2(1,0)).r;
    float t2 = imageLoad(GetUAVImage2DR32F(MipId), MipUV+ivec2(0,1)).r;
    float t3 = imageLoad(GetUAVImage2DR32F(MipId), MipUV+ivec2(1,1)).r;
    return min(min(t0,t1),min(t2,t3));
}

float GetHizDepth(vec2 UV, uint Mip){
    vec2 PixelUV = UV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    ivec2 PixelUVInt = ivec2(PixelUV);
    return GetHizDepth(PixelUVInt, Mip, false);
}

vec3 SsgiTraceImpl(vec3 RayStartVS, vec3 RayEndVS, vec2 RayStartUV, vec2 RayEndUV){
    PerFramePerViewData PerFrame = GetResource(BPerframe,PushConst.m_PerFrameCBV).m_Data;
    float ClipNear = PerFrame.m_cameraNear;
    float ClipFar = PerFrame.m_cameraFar;

    vec2 DiffUV = RayEndUV - RayStartUV;
    vec2 DiffPixels = DiffUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    ivec2 DiffPixelsInt = ivec2(DiffPixels);
    ivec2 RayStartUVInt = ivec2(RayStartUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight));

    float MaxStepsF = max(abs(DiffPixels.x), abs(DiffPixels.y));
    uint MaxSteps = uint(MaxStepsF) + 1; // total steps required on marching Hiz level 0
    float MinimalStep = 1.0 / float(MaxSteps);
    
    int CurMip = 0;
    bool FinalHit = true;
    vec2 HitUV = vec2(-1.0,0.0);
    float DepthDiffVS = 0.0;

    int ProceedSignX = DiffPixels.x > 0.0 ? 1 : -1;
    int ProceedSignY = DiffPixels.y > 0.0 ? 1 : -1;
    float CurStepF  = 0.0;
    int MaxIters = int(kMaxTraceIters);
    int CurIters = 0;

    while(CurMip >= 0 && CurIters < MaxIters){
        CurIters += 1;
        float T = CurStepF;
        vec2 CurUV = mix(RayStartUV, RayEndUV, T);
        float ReferenceZ = GetHizDepth(CurUV, CurMip);
        bool ValidZ = ReferenceZ > 0.0 && ReferenceZ < 1.0;
        ReferenceZ = ifrit_recoverViewSpaceDepth(ReferenceZ, ClipNear, ClipFar);
        float CurZ = ifrit_perspectiveLerp(RayStartVS.z, RayEndVS.z, RayStartVS.z, RayEndVS.z, T);
        ivec2 CurUVInt = ivec2(CurUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight));
        ivec2 CurUVIntMip = CurUVInt >> CurMip;

        bool IsCollided = false;
        if(CurZ >= ReferenceZ && (CurMip!=0 || ValidZ)){
            IsCollided = true;
        }

        if(IsCollided && CurMip == 0){
            FinalHit = true;
            HitUV = mix(RayStartUV, RayEndUV, T);
            DepthDiffVS = CurZ - ReferenceZ;
            break;
        }

        // step if not collided
        if(!IsCollided){
            int NextTexelX = ((CurUVIntMip.x + ProceedSignX)<<CurMip) - RayStartUVInt.x;
            int NextTexelY = ((CurUVIntMip.y + ProceedSignY)<<CurMip) - RayStartUVInt.y;

            float StepX = float(NextTexelX) / float(DiffPixelsInt.x);
            float StepY = float(NextTexelY) / float(DiffPixelsInt.y);
            float NextStep = max(CurStepF, min(abs(StepX), abs(StepY)));
            CurStepF = NextStep;
            if(kHizProceed)
                CurMip = min(CurMip+1, 6);
        }else{
            if(kHizProceed)
                CurMip-=1;
        }
    }

    // Check the hit z difference
    if(abs(DepthDiffVS) > 0.25){
        //FinalHit = false;
    }
    return vec3(HitUV, FinalHit ? 1.0 : 0.0);
}

vec3 SsgiRayTrace(vec3 RayOriginWS, vec3 RayDirWS){
    PerFramePerViewData PerFrame = GetResource(BPerframe,PushConst.m_PerFrameCBV).m_Data;
    bool ValidSample = true;
    mat4 WorldToClip = PerFrame.m_worldToClip;
    mat4 ClipToWorld = PerFrame.m_clipToWorld;
    mat4 WorldToView = PerFrame.m_worldToView;
    mat4 ViewToClip = PerFrame.m_perspective;
    float ClipNear = PerFrame.m_cameraNear;

    vec4 OriginVS = WorldToView * vec4(RayOriginWS, 1.0);
    vec4 ProceedVS = WorldToView * vec4(RayOriginWS + RayDirWS * kRayProceedMax, 1.0);
    vec4 RayDirVS = ProceedVS - OriginVS;
    // Need to limit the proceed vs in the view frustum. (at least, larger than near plane)
    if(ProceedVS.z < ClipNear+1e-3){
        // find O+td intersection with near plane.
        float t = (ClipNear+1e-3 - OriginVS.z) / RayDirVS.z;
        if(abs(RayDirVS.z) < 1e-4){
           ValidSample = false;
        }
        ProceedVS = OriginVS + RayDirVS * t;
    }

    vec4 OriginCS = ViewToClip * OriginVS;
    vec4 ProceedCS = ViewToClip * ProceedVS;

    vec2 OriginNDCxy = OriginCS.xy / OriginCS.w;
    vec2 ProceedNDCxy = ProceedCS.xy / ProceedCS.w;
    vec2 OriginUV = (OriginNDCxy + 1.0) * 0.5;
    vec2 ProceedUV = (ProceedNDCxy + 1.0) * 0.5;

    vec2 ProceedDirUV = ProceedUV - OriginUV;
    vec2 RayNDCIntersection = ifrit_RayIntersectWithUnitRect2D(OriginUV, ProceedDirUV);

    // The tracing center does not present in the screen space.
    if(OriginUV.x < 0.0 || OriginUV.x > 1.0 || OriginUV.y < 0.0 || OriginUV.y > 1.0){
        ValidSample = false;
    }
    if(RayNDCIntersection.x>0.0 || RayNDCIntersection.y < 0.0){
        ValidSample = false;
    }
    vec2 AbsProceedDirUV = abs(ProceedDirUV);
    if(AbsProceedDirUV.x < 1e-4 || AbsProceedDirUV.y < 1e-4){
        ValidSample = false;
    }

    vec3 RayTraceStartVS = OriginVS.xyz;
    vec3 RayTraceEndVS = ProceedVS.xyz;
    vec2 RayTraceStartUV = OriginUV;
    vec2 RayTraceEndUV = ProceedUV;

    vec3 RayTraceDirNew = normalize(RayTraceEndVS - RayTraceStartVS);
    vec3 RayTraceDirOld = normalize(ProceedVS.xyz - OriginVS.xyz);
    
    if(!ValidSample){
        // The ray is not valid, return the invalid color
        return vec3(0.0, 0.0, 0.0);
    }
    return SsgiTraceImpl(RayTraceStartVS, RayTraceEndVS, RayTraceStartUV, RayTraceEndUV);
}

void main(){
    uvec2 ScreenCoord = uvec2(gl_GlobalInvocationID.xy);
    if(ScreenCoord.x >= PushConst.m_RTWidth || ScreenCoord.y >= PushConst.m_RTHeight) return;

    PerFramePerViewData PerFrame = GetResource(BPerframe,PushConst.m_PerFrameCBV).m_Data;

    vec2 ScreenUV = (vec2(ScreenCoord) + vec2(0.5)) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    float DepthCS = GetHizDepth(ScreenUV, 0);
    vec3 ScreenNDC = vec3(ScreenUV*2.0 - 1.0, DepthCS);

    mat4 InvPerspective = PerFrame.m_invPerspective;
    mat4 ClipToWorld = PerFrame.m_clipToWorld;
    vec4 LocationVS = InvPerspective * vec4(ScreenNDC, 1.0);
    vec4 LocationWS = ClipToWorld * vec4(ScreenNDC, 1.0);
    LocationWS /= LocationWS.w;
    LocationVS /= LocationVS.w;

    vec3 NormalVS = texelFetch(GetSampler2D(PushConst.m_NormalTexSRV), ivec2(ScreenCoord), 0).xyz;
    NormalVS = normalize(NormalVS * 2.0 - 1.0);
    vec3 NormalWS = (PerFrame.m_worldToView * vec4(NormalVS, 0.0)).xyz;
    NormalWS = normalize(NormalWS);

    vec3 InRay = normalize(LocationVS.xyz);
    float NumValidSamples = 0.0f;
    MThreeBandSH_RGB SHCoefs = ifrit_ZeroSH3RGB();

    for(uint S=0;S<cSSGISamples;S++){
        uint RayCoordX = S % cSSGIHemiProbeHemiRes;
        uint RayCoordY = S / cSSGIHemiProbeHemiRes;
        uvec2 RayCoord = uvec2(RayCoordX, RayCoordY);   
        vec2 RayUV = (vec2(RayCoord) + vec2(0.5)) / vec2(cSSGIHemiProbeHemiRes);

        vec4 SampledRayAndPDF = ifrit_SampleCosineHemisphereWithPDF(RayUV, NormalWS);
        vec3 SampledRay = SampledRayAndPDF.xyz;
        float SampledPDF = SampledRayAndPDF.w;

        vec3 TraceLocationWS = LocationWS.xyz + SampledRay * 0.1;
        
        vec3 TracingResultRaw = SsgiRayTrace(TraceLocationWS, SampledRay);
        vec4 TracingResult;
        TracingResult.w = TracingResultRaw.z;

        vec2 HitUV = vec2(TracingResultRaw.x, TracingResultRaw.y);
        vec3 HitLighting = SampleTexture2D(PushConst.m_FinalLightingSRV,sLinearClamp,HitUV).rgb;//texture(GetSampler2D(PushConst.m_FinalLightingSRV), HitUV).xyz;
        if(HitUV.x == -1.0 && HitUV.y == 0.0){
            HitLighting = vec3(0.0,0.0,0.0); //skylight, for simplicity
        }
        TracingResult.xyz = HitLighting;

        if(TracingResult.w > 0.0){
            NumValidSamples += 1.0;
            MThreeBandSH_RGB SHBasis = ifrit_SHBasis3EncodeRGB(SampledRay);
            MThreeBandSH_RGB SHResult = ifrit_MulSH3RGBColor(SHBasis, TracingResult.xyz / SampledPDF);

            SHCoefs = ifrit_AddSH3RGB(SHCoefs, SHResult);
        }
    }
    if(NumValidSamples > 0.0){
        SHCoefs = ifrit_MulSH3RGB(SHCoefs, 1.0 / NumValidSamples);
    }

    MThreeBandSH DiffuseTransfer = ifrit_SHCosineLobe3Encode(NormalWS);
    float DiffuseLobeR = ifrit_DotSH3(DiffuseTransfer, SHCoefs.m_R);
    float DiffuseLobeG = ifrit_DotSH3(DiffuseTransfer, SHCoefs.m_G);
    float DiffuseLobeB = ifrit_DotSH3(DiffuseTransfer, SHCoefs.m_B);

    vec3 Albedo = texelFetch(GetSampler2D(PushConst.m_AlbedoTexSRV), ivec2(ScreenCoord), 0).xyz;

    vec3 LambertBRDF = Albedo * (1.0 / 3.14159265358979323846);
    vec3 Irradiance = vec3(DiffuseLobeR, DiffuseLobeG, DiffuseLobeB);
    Irradiance = max(Irradiance, vec3(0.0)) ;
    Irradiance = Irradiance * LambertBRDF;

    // Store GI
    vec4 AoRaw = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_AOTexUAV), ivec2(ScreenCoord));
    AoRaw.xyz = Irradiance;
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_AOTexUAV), ivec2(ScreenCoord), AoRaw);
}
