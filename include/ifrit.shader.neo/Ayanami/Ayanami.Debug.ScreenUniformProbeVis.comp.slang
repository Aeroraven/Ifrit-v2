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
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"


namespace IfritShader{
namespace Ayanami{

    struct PushConstant_T
    {
        uint m_RTWidth;
        uint m_RTHeight;
        TTexture2DHandle<float4> m_Albedo;
        TRWTexture2DHandle<float4> m_Output;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiDbgScrProbeUniformVisKernelSize,
        kAyanamiDbgScrProbeUniformVisKernelSize,
        1
    )
    void DebugScreenUniformProbeVisCS(uint3 ThreadID: SV_DispatchThreadID,
                                        uint3 GroupID: SV_GroupID,
                                        uint3 GroupThreadID: SV_GroupThreadID)
    {
        bool Valid = true;
        if(ThreadID.x >= PushConst.m_RTWidth || ThreadID.y >= PushConst.m_RTHeight){
            Valid=false;
        }
        bool IsProbeX = ThreadID.x % kAyanami_ScreenProbeUniformPlaceTileWidth == 0;
        bool IsProbeY = ThreadID.y % kAyanami_ScreenProbeUniformPlaceTileWidth == 0;

        if(IsProbeX && IsProbeY){
            PushConst.m_Output.Store(ThreadID.xy,float4(1.0, 0.0, 0.0, 1.0));
            
        }else{
            float2 UV = (float2(ThreadID.xy) + float2(0.5)) / float2(PushConst.m_RTWidth, PushConst.m_RTHeight);
            float3 Albedo = PushConst.m_Albedo.SampleLevel(ESamplerType::SLinearClamp, UV, 0.0f).xyz;
            PushConst.m_Output.Store(ThreadID.xy, float4(Albedo, 1.0));
        }
    }
            

}
}