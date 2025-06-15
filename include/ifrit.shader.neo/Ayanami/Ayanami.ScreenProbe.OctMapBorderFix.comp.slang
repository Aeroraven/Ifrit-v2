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

namespace IfritShader{
namespace Ayanami{

    struct PushConstant_T
    {
        FAdaptiveProbesCounterHandle m_AdaptiveProbeCounter;
        uint m_RTWidth;
        uint m_RTHeight;
        TRWTexture2DHandle<float4> m_ScreenProbeLightingAtlasIn;
        TRWTexture2DHandle<float4> m_ScreenProbeLightingAtlasOut;
    };

    uint2 GetProbeWritingSlotReadIn(uint ProbeId, uint ProbeCntPerX, uint2 TraceRayCoord){
        uint ProbeX = ProbeId % ProbeCntPerX;
        uint ProbeY = ProbeId / ProbeCntPerX;
        uint ProbeLocX = ProbeX * kAyanami_ScreenProbeProbeHemiRes;
        uint ProbeLocY = ProbeY * kAyanami_ScreenProbeProbeHemiRes;
        uint2 ProbeLoc = uint2(ProbeLocX, ProbeLocY);
        uint2 WritingSlot = ProbeLoc + TraceRayCoord;
        return WritingSlot; 
    }
    
    uint2 GetProbeWritingSlotWriteTo(uint ProbeId, uint ProbeCntPerX, uint2 TraceRayCoord){
        uint ProbeX = ProbeId % ProbeCntPerX;
        uint ProbeY = ProbeId / ProbeCntPerX;
        uint ProbeLocX = ProbeX * (kAyanami_ScreenProbeProbeHemiRes+2);
        uint ProbeLocY = ProbeY * (kAyanami_ScreenProbeProbeHemiRes+2);
        uint2 ProbeLoc = uint2(ProbeLocX, ProbeLocY);
        uint2 WritingSlot = ProbeLoc + TraceRayCoord;
        return WritingSlot; 
    }

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiScrProbeIntegrateSHKernelSize,
        1,
        1
    )
    void ScreenProbeOctMapBorderFixCS(uint3 ThreadID : SV_DispatchThreadID,
                                    uint3 GroupID : SV_GroupID,
                                    uint3 GroupThreadID : SV_GroupThreadID)
    {
        uint ProbeCntPerX = DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint ProbeCntPerY = DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;
        uint TotalProbes = PushConst.m_AdaptiveProbeCounter.GetCounter() + TotalUniformProbes;

        uint ProbeId = ThreadID.x;
        if(ProbeId >= TotalProbes) return;

        for(uint i=0;i<kAyanami_ScreenProbeProbeHemiRes;i++){
            for(uint j=0;j<kAyanami_ScreenProbeProbeHemiRes;j++){
                uint2 TraceRayCoord = uint2(i,j);
                uint2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, TraceRayCoord);
                uint2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, TraceRayCoord+uint2(1,1));
                float3 Irradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(WritingSlotIn).xyz;
                PushConst.m_ScreenProbeLightingAtlasOut.Store(WritingSlotOut, float4(Irradiance, 1.0f));
            }
        }
    
        // Fixing Borders
        uint LocX=0;
        uint LocY=0;

        for(LocX=1;LocX<=kAyanami_ScreenProbeProbeHemiRes;LocX++){
            // Output:(LocX,0) <= Input:(N-LocX,0)
            uint2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uint2(kAyanami_ScreenProbeProbeHemiRes-LocX,0));
            uint2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uint2(LocX,0));
            float3 Irradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(WritingSlotIn).xyz;
            PushConst.m_ScreenProbeLightingAtlasOut.Store(WritingSlotOut, float4(Irradiance, 1.0f));
        }
        // left
        for(LocY=1;LocY<=kAyanami_ScreenProbeProbeHemiRes;LocY++){
            // Output:(0,LocY) <= Input:(0,N-LocY)
            uint2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uint2(0,kAyanami_ScreenProbeProbeHemiRes-LocY));
            uint2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uint2(0,LocY));
            float3 Irradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(WritingSlotIn).xyz;
            PushConst.m_ScreenProbeLightingAtlasOut.Store(WritingSlotOut, float4(Irradiance, 1.0f));
    }
        // bottom
        for(LocX=1;LocX<=kAyanami_ScreenProbeProbeHemiRes;LocX++){
            // Output:(LocX,N+1) <= Input:(N-LocX,N+1)
            uint2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uint2(kAyanami_ScreenProbeProbeHemiRes-LocX,kAyanami_ScreenProbeProbeHemiRes-1));
            uint2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uint2(LocX,kAyanami_ScreenProbeProbeHemiRes+1));
            float3 Irradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(WritingSlotIn).xyz;
            PushConst.m_ScreenProbeLightingAtlasOut.Store(WritingSlotOut, float4(Irradiance, 1.0f));
        }
        // right
        for(LocY=1;LocY<=kAyanami_ScreenProbeProbeHemiRes;LocY++){
            // Output:(N+1,LocY) <= Input:(N+1,N-LocY)
            uint2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uint2(kAyanami_ScreenProbeProbeHemiRes-1,kAyanami_ScreenProbeProbeHemiRes-LocY));
            uint2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uint2(kAyanami_ScreenProbeProbeHemiRes+1,LocY));
            float3 Irradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(WritingSlotIn).xyz;
            PushConst.m_ScreenProbeLightingAtlasOut.Store(WritingSlotOut, float4(Irradiance, 1.0f));
        }
        
        // top-left
        uint2 CornerSlotIn;
        uint2 CornerSlotOut;
        float3 CornerIrradiance;

        CornerSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uint2(0,0));
        CornerSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uint2(kAyanami_ScreenProbeProbeHemiRes+1,kAyanami_ScreenProbeProbeHemiRes+1));
        CornerIrradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(CornerSlotIn).xyz;
        PushConst.m_ScreenProbeLightingAtlasOut.Store(CornerSlotOut, float4(CornerIrradiance, 1.0f));
        // top-right
        CornerSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uint2(kAyanami_ScreenProbeProbeHemiRes-1,0));
        CornerSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uint2(0,kAyanami_ScreenProbeProbeHemiRes+1));
        CornerIrradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(CornerSlotIn).xyz;
        PushConst.m_ScreenProbeLightingAtlasOut.Store(CornerSlotOut, float4(CornerIrradiance, 1.0f));

        // bottom-left
        CornerSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uint2(0,kAyanami_ScreenProbeProbeHemiRes-1));
        CornerSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uint2(kAyanami_ScreenProbeProbeHemiRes+1,0));
        CornerIrradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(CornerSlotIn).xyz;
        PushConst.m_ScreenProbeLightingAtlasOut.Store(CornerSlotOut, float4(CornerIrradiance, 1.0f));

        // bottom-right
        CornerSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uint2(kAyanami_ScreenProbeProbeHemiRes-1,kAyanami_ScreenProbeProbeHemiRes-1));
        CornerSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uint2(0,0));
        CornerIrradiance = PushConst.m_ScreenProbeLightingAtlasIn.Load(CornerSlotIn).xyz;
        PushConst.m_ScreenProbeLightingAtlasOut.Store(CornerSlotOut, float4(CornerIrradiance, 1.0f));

    }
    

} // namespace Ayanami
} // namespace IfritShader