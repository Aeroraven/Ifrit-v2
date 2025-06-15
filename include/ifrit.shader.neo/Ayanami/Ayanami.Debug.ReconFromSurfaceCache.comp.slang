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

#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"

namespace IfritShader{
namespace Ayanami{

    struct PushConstant_T
    {
        uint m_NumTotalCards;
        uint m_CardResolution;
        uint m_CardAtlasResolution;
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        TRWStructuredBufferHandle<CardTransformData> m_CardData;
        uint m_Dummy;
        TTexture2DHandle<float4> m_CardAlbedoAtlas;
        TTexture2DHandle<float4> m_CardNormalAtlas;
        TTexture2DHandle<float4> m_CardRadianceAtlas;
        TTexture2DHandle<float> m_CardDepthAtlas;
        TAtomicRWStructuredBufferHandle<uint64_t> m_AtomicDepth;
        uint m_OutputWidth;
        uint m_OutputHeight;
        TRWStructuredBufferHandle<FMeshDFDescriptor> m_MeshDFDesc;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    void WritePixel(uint2 AtlasUV, float Depth, uint2 ScreenUV){
        // 32Bit depth | 16Bit AtlasX | 16Bit AtlasY 
        uint64_t DepthData = uint(Depth * 65535.0) << 32 | (AtlasUV.x << 16) | AtlasUV.y;
        uint LinearizedId = ScreenUV.y * PushConst.m_OutputWidth + ScreenUV.x;
        PushConst.m_AtomicDepth.AtomicMin(LinearizedId, DepthData);
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiReconFromSCDepthTileSize,
        kAyanamiReconFromSCDepthTileSize,
        1
    )
    void DebugReconFromSurfaceCacheCS(uint3 ThreadID: SV_DispatchThreadID,
                                         uint3 GroupID: SV_GroupID,
                                         uint3 GroupThreadID: SV_GroupThreadID)
    {
        uint CardId = ThreadID.z;
        uint TileX = ThreadID.x;
        uint TileY = ThreadID.y;

        uint CardsPerRow = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
        uint CardX = CardId % CardsPerRow;
        uint CardY = CardId / CardsPerRow;
        uint2 TileCardOffset = uint2(CardX, CardY) * PushConst.m_CardResolution + uint2(TileX, TileY);

        float2 AtlasUV = (float2(TileCardOffset) + float2(0.5)) / float2(PushConst.m_CardAtlasResolution);
        float Depth = PushConst.m_CardDepthAtlas.SampleLevel(ESamplerType::SLinearClamp, AtlasUV, 0.0f);
        if(Depth == 1.0) return;

        float2 CardUV = (float2(TileX, TileY) + float2(0.5)) / float2(PushConst.m_CardResolution);
        float4 CardNDC  = float4(CardUV * 2.0 - 1.0, Depth, 1.0);

        float4x4 CardToLocal = PushConst.m_CardData.Load(CardId).m_VPInv;
        float4 LocalPos = mul(CardToLocal, CardNDC);
        if(LocalPos.w <= 0.0) return;
        LocalPos /= LocalPos.w;
        
        FMeshDFDescriptor MDFDesc = PushConst.m_MeshDFDesc.Load(CardId / 6);
        
        float4x4 LocalToWorld = MDFDesc.m_Transform.Load().m_LocalToWorld;
        float4x4 WorldToScreen = PushConst.m_PerFrame.Load().m_WorldToClip;
        float4x4 LocalToScreen  = mul(WorldToScreen , LocalToWorld);

        float4 ClipPos = mul(LocalToScreen, LocalPos);
        if(ClipPos.w <= 0.0) return;
        ClipPos /= ClipPos.w;
        
        float2 ScreenUV = ClipPos.xy * 0.5 + 0.5;
        if(ScreenUV.x < 0.0 || ScreenUV.x > 1.0 || ScreenUV.y < 0.0 || ScreenUV.y > 1.0) return;

        uint2 ScreenUVi = uint2(ScreenUV * float2(PushConst.m_OutputWidth, PushConst.m_OutputHeight));

        WritePixel(uint2(TileCardOffset.x, TileCardOffset.y), Depth, ScreenUVi);
    }
}
}