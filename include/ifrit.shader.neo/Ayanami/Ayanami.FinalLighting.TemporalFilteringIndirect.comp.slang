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

namespace IfritShader
{
    struct PushConstant_T
    {
        uint m_FrameIndex;
        TRWTexture2DHandle<float4> m_CurrentFrameHandle;
        TRWTexture2DHandle<float4> m_PreviousFrameHandle;
        uint m_Width;
        uint m_Height;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiFinalTemporalReprojKernelSizeX,
        kAyanamiFinalTemporalReprojKernelSizeY,
        1
    )
    void FinalLightingTemporalFilteringIndirectCS(
        uint3 ThreadId : SV_DispatchThreadID
    )
    {
        if(ThreadId.x >= PushConst.m_Width || ThreadId.y >= PushConst.m_Height)
        {
            return; // Out of bounds
        }
        float4 HistoryColor = PushConst.m_PreviousFrameHandle.Load(ThreadId.xy);
        float4 CurrentColor = PushConst.m_CurrentFrameHandle.Load(ThreadId.xy);

        uint NumHistorySamples = PushConst.m_FrameIndex;
        if(NumHistorySamples > 0)
        {
            NumHistorySamples -= 1; 
        }
        float4 MixedColor = lerp(HistoryColor, CurrentColor, 1.0f / (NumHistorySamples + 1));
        PushConst.m_CurrentFrameHandle.Store(ThreadId.xy, MixedColor);
    }
    
}