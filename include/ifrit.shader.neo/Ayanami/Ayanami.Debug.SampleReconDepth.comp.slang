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

namespace IfritShader{
namespace Ayanami{

    struct PushConstant_T
    {
        uint m_NumTotalCards;
        uint m_CardResolution;
        uint m_CardAtlasResolution;
        uint m_PerFrameDataId;
        uint m_CardDataBufferId;
        TRWTexture2DHandle<float4> m_Output;
        TTexture2DHandle<float4> m_CardAlbedoAtlas;
        TTexture2DHandle<float4> m_CardNormalAtlas;
        TTexture2DHandle<float4> m_CardRadianceAtlas;
        TTexture2DHandle<float> m_CardDepth;;
        //TRWTexture2DHandle<uint64_t> m_AtomicDepth; , slang does not support this now >///<
        TRWStructuredBufferHandle<uint64_t> m_AtomicDepth;
        uint m_OutputWidth;
        uint m_OutputHeight;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiReconFromSCTileSize,
        kAyanamiReconFromSCTileSize,
        1
    )
    void DebugSampleReconDepthCS(uint3 ThreadID: SV_DispatchThreadID,
                                 uint3 GroupID: SV_GroupID,
                                 uint3 GroupThreadID: SV_GroupThreadID)
    {
        if (ThreadID.x >= PushConst.m_OutputWidth || ThreadID.y >= PushConst.m_OutputHeight)
            return;

        uint LinearizedId = ThreadID.y * PushConst.m_OutputWidth + ThreadID.x;

        uint64_t DepthValue = PushConst.m_AtomicDepth.Load(LinearizedId);
        uint64_t DepthBits = DepthValue >> 32;
        uint32_t U32Max = 0xFFFFFFFF;
        float Depth = float(DepthBits) / float(U32Max);

        if(Depth>=0.99)
        {
            PushConst.m_Output.Store(uint2(ThreadID.xy), float4(0, 0, 0, 1));
            return;
        }

        uint2 AtlasUV = uint2(uint(DepthValue & uint64_t(0xFFFF)), uint((DepthValue >> 16) & 0xFFFF)).yx;
        float2 AtlasUVf = (float2(AtlasUV)+0.5f) / float(PushConst.m_CardAtlasResolution);
        
        if(AtlasUVf.x >= 1.0f || AtlasUVf.y >= 1.0f)
        {
            return ;
        }
        float3 Albedo = PushConst.m_CardAlbedoAtlas.SampleLevel(ESamplerType::SLinearClamp, AtlasUVf, 0.0f).xyz;
        PushConst.m_Output.Store(uint2(ThreadID.xy), float4(Albedo, 1.0f));
    }

}
}