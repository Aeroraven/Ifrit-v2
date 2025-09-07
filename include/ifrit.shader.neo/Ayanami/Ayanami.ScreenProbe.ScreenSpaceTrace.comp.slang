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
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"
#include "ifrit.shader.neo/SSGI/ScreenSpaceTrace.hlsli"

namespace IfritShader {
namespace Ayanami {

    struct PushConstant_T
    {
        float2 m_RayJitter;
        HzbDataHandle m_HizStorage;
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        uint m_RtWidth;
        uint m_RtHeight;
        TRWTexture2DHandle<float4> m_ScreenProbeLightingAtlas;
        FMeshDFTraceProposalIndirectArgsHandle m_MDFProposalCounter;
        TRWStructuredBufferHandle<uint> m_MDFProposalList;
        TRWStructuredBufferHandle<uint> m_AdaptiveProbeList;
        TTexture2DHandle<float4> m_LastFrameLighting;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    static const float kRayProceedAdvance = 6e-3f;
    static const uint kMaxTraceIters = 60;

    groupshared uint sFailureRayCount;
    groupshared uint sFailureRayGlobalStart;
    groupshared uint sFailureRayList[kAyanami_ScreenProbeProbeHemiRes * kAyanami_ScreenProbeProbeHemiRes];

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiScrProbeScreenTraceKernelSize,
        kAyanamiScrProbeScreenTraceKernelSize,
        1
    )
    void ScreenProbeScreenSpaceTraceCS(uint3 ThreadID: SV_DispatchThreadID,
                                       uint3 GroupID: SV_GroupID,
                                       uint3 GroupThreadID: SV_GroupThreadID)
    {
        if(IsFirstLane(GroupThreadID))
        {
            sFailureRayCount = 0;
        }
        if(IsGlobalFirstThread(ThreadID))
        {
            PushConst.m_MDFProposalCounter.WriteInvoY(1);
            PushConst.m_MDFProposalCounter.WriteInvoZ(1);
        }
        GroupMemoryBarrierWithGroupSync();

        uint ProbeId = GroupID.x;
        uint2 TraceRayCoord = GroupThreadID.xy;

        PerFramePerViewData PerFrame = PushConst.m_PerFrame.Load();

        float ClipNear = PerFrame.m_CameraNear;
        float ClipFar = PerFrame.m_CameraFar;
        float4x4 ClipToWorld = PerFrame.m_ClipToWorld;

        float3 SampledRay = GetScreenProbeTraceCoord(TraceRayCoord,PushConst.m_RayJitter);

        uint ProbeCntPerX = DivRoundUp(PushConst.m_RtWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint ProbeCntPerY = DivRoundUp(PushConst.m_RtHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;

        float3 ProbeLocWS;
        float2 ProbeUV;
        float2 ProbeUVPx;
        if(ProbeId < TotalUniformProbes){
            uint ProbeX = ProbeId % ProbeCntPerX;
            uint ProbeY = ProbeId / ProbeCntPerX;
            uint ProbeLocX = ProbeX * kAyanami_ScreenProbeUniformPlaceTileWidth;
            uint ProbeLocY = ProbeY * kAyanami_ScreenProbeUniformPlaceTileWidth;
            ProbeUV = float2(ProbeLocX, ProbeLocY) / float2(PushConst.m_RtWidth, PushConst.m_RtHeight);
            ProbeUVPx = float2(ProbeLocX, ProbeLocY);
        }else{
            // This is an adaptive probe
            uint AdaptiveProbeId = ProbeId - TotalUniformProbes;
            uint2 AdaptiveProbeCoord = UnpackAdaptiveProbeLocation(PushConst.m_AdaptiveProbeList.Load(AdaptiveProbeId));
            ProbeUV = float2(AdaptiveProbeCoord) / float2(PushConst.m_RtWidth, PushConst.m_RtHeight);
            ProbeUVPx = float2(AdaptiveProbeCoord);
        }

        float ProbeLocDepthNDC = PushConst.m_HizStorage.GetPixel(0, uint2(ProbeUVPx));
        bool ValidProbe = true;
        if(ProbeLocDepthNDC >= 1.0){
            ValidProbe = false;
        }

        float ProbeLocDepthVS = Math::ClipDepthToViewDepth(ProbeLocDepthNDC, ClipNear, ClipFar);
        float3 ProbeLocNDC = float3(ProbeUV * 2.0-1.0, ProbeLocDepthNDC);
        float3 ProbeLocCS = ProbeLocNDC * ProbeLocDepthVS;
        float4 ProbeLocWSH = mul(ClipToWorld , float4(ProbeLocCS, ProbeLocDepthVS));
        ProbeLocWS = ProbeLocWSH.xyz / ProbeLocWSH.w;
        ProbeLocWS += SampledRay * kRayProceedAdvance;
        
        SSGI::ScreenSpaceTraceResult SsgiTraceResult;
        SsgiTraceResult.FinalHit = false;
        SsgiTraceResult.HitUV = float2(0.0f, 0.0f);
        if(ValidProbe)
        {
            SsgiTraceResult = SSGI::ScreenSpaceTrace(
                SampledRay,
                ProbeLocWS,
                PushConst.m_PerFrame,
                PushConst.m_HizStorage,
                kMaxTraceIters,
                float2(PushConst.m_RtWidth, PushConst.m_RtHeight),
            );
        }

        uint2 WritingSlot = GetScreenSpaceProbeWritingSlot(ProbeId, ProbeCntPerX, TraceRayCoord);

        // Writing probe
        if(ValidProbe)
        {
            if(SsgiTraceResult.FinalHit)
            {
                float3 HitIrradiance = PushConst.m_LastFrameLighting.SampleLevel(ESamplerType::SLinearClamp, SsgiTraceResult.HitUV, 0.0f).xyz;
                PushConst.m_ScreenProbeLightingAtlas.Store(WritingSlot, float4(HitIrradiance, 1.0f));
            }
            else
            {
                PushConst.m_ScreenProbeLightingAtlas.Store(WritingSlot, float4(0.0f, 0.0f, 0.0f,  ProbeLocDepthNDC));
                uint FailureRayIndex;
                InterlockedAdd(sFailureRayCount, 1, FailureRayIndex);
                sFailureRayList[FailureRayIndex] = ScreenProbePackLocationAndRay(ProbeId, TraceRayCoord);
            }
        }
        else
        {
            PushConst.m_ScreenProbeLightingAtlas.Store(WritingSlot, float4(0.0f, 0.0f, 0.0f,  ProbeLocDepthNDC));     
        }
        GroupMemoryBarrierWithGroupSync();

        if(IsFirstLane(GroupThreadID))
        {
            uint FailureRayCount = sFailureRayCount;
            uint GlobalListStart = PushConst.m_MDFProposalCounter.AddFailureCount(FailureRayCount);
            uint TotalFailureRayCount = GlobalListStart + FailureRayCount;
            uint TotalMDFTraceTGs = DivRoundUp(TotalFailureRayCount, kAyanamiScrProbeMDFTraceKernelSize);
            sFailureRayGlobalStart = GlobalListStart;
            PushConst.m_MDFProposalCounter.WriteInvoXAtomic(TotalMDFTraceTGs);
        }
        
        GroupMemoryBarrierWithGroupSync();
        uint LocalId = GroupThreadID.x + GroupThreadID.y * kAyanamiScrProbeScreenTraceKernelSize;
        if(LocalId < sFailureRayCount)
        {
            uint FailureRayId = sFailureRayList[LocalId];
            uint GlobalRayId = sFailureRayGlobalStart + LocalId;
            PushConst.m_MDFProposalList.Store(FailureRayId,GlobalRayId);
        }
    }   
    
}
}