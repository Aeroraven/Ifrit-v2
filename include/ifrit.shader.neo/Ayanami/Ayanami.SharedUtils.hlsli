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
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedConst.hlsli"
#include "ifrit.shader.neo/Math.MonteCarlo.hlsli"
#include "ifrit.shader.neo/Math.Transforms.hlsli"

namespace IfritShader{
namespace Ayanami{

    #define INTERNAL_AYANAMI_RADIOSITY_TILE_SHCONV_DEBUG 1

    struct CardTransformData
    {
        float4x4 m_VP;
        float4x4 m_VPInv;
    };

    struct FMeshDFTraceProposalIndirectArgs
    {
        uint m_NumMDFProposals;
        uint m_InvoX;
        uint m_InvoY;
        uint m_InvoZ;
    };

    struct FAdaptiveProbesCounter
    {
        uint m_Counter;
        uint m_InvoX; // below two are used for debugging only
        uint m_InvoY; 
        uint m_InvoZ;
        uint m_InvoTraceX; // used for screen space tracing
        uint m_InvoTraceY;
        uint m_InvoTraceZ;
    };

    struct FMeshDFDetail{
        float4 m_BBoxMin;
        float4 m_BBoxMax;
        TTexture3DHandle<float> m_SdfId;
        uint m_IsTwoSided;

        float2 GetSdfQuantScale()
        {
            return float2(m_BBoxMin.w, m_BBoxMax.w);
        }

        float SampleSdf(float3 UVW)
        {
            float SdfValue = m_SdfId.SampleLevel(ESamplerType::SLinearClamp,UVW,0.0f);
            float2 QuantScale = GetSdfQuantScale();
            return lerp(QuantScale.x, QuantScale.y, SdfValue);
        }
    };

    struct FMeshDFDescriptor
    {
        TRWStructuredBufferHandle<FMeshDFDetail> m_MdfDetail;
        TConstantBufferHandle<FInstanceLocalTransform> m_Transform;
    };

    struct FMeshDFTraceProposalIndirectArgsHandle
    {
        TAtomicRWStructuredBufferHandle<uint> m_IndirectArgs;

        void WriteInvoX(uint InvoX)
        {
            m_IndirectArgs.Store(InvoX,1);
        }

        void WriteInvoY(uint InvoY)
        {
            m_IndirectArgs.Store(InvoY,2);
        }

        void WriteInvoZ(uint InvoZ)
        {
            m_IndirectArgs.Store(InvoZ,3);
        }

        uint AddFailureCount(uint Count)
        {
            return m_IndirectArgs.AtomicAdd(0, Count);
        }

        void WriteInvoXAtomic(uint InvoX)
        {
            m_IndirectArgs.AtomicMax(1, InvoX);
        }
    }

    struct FAdaptiveProbesCounterHandle
    {
        TAtomicRWStructuredBufferHandle<uint> m_AdaptiveProbesCounter;
        uint GetCounter()
        {
            return m_AdaptiveProbesCounter.Load(0);
        }
    };


    float3 GetScreenProbeTraceCoord(uint2 TraceRayCoord, float2 Jitter){
        float2 UV = (float2(TraceRayCoord)+Jitter + float2(0.5f)) / float2(kAyanami_ScreenProbeProbeHemiRes);
        return Math::ConcentricOctahedralTransform(UV).xyz;
    }

    uint2 UnpackAdaptiveProbeLocation(uint PackedCoords){
        uint y = PackedCoords & 0xFFFF;
        uint x = (PackedCoords >> 16) & 0xFFFF;
        return uint2(x,y);
    }

    uint2 GetScreenSpaceProbeWritingSlot(uint ProbeId, uint ProbeCntPerX, uint2 TraceRayCoord){
        uint ProbeX = ProbeId % ProbeCntPerX;
        uint ProbeY = ProbeId / ProbeCntPerX;
        uint ProbeLocX = ProbeX * kAyanami_ScreenProbeProbeHemiRes;
        uint ProbeLocY = ProbeY * kAyanami_ScreenProbeProbeHemiRes;
        uint2 ProbeLoc = uint2(ProbeLocX, ProbeLocY);
        uint2 WritingSlot = ProbeLoc + TraceRayCoord;
        return WritingSlot; 
    }

    uint2 GetScreenSpaceProbeWritingSlotAfterFix(uint ProbeId, uint ProbeCntPerX, uint2 TraceRayCoord){
        uint ProbeX = ProbeId % ProbeCntPerX;
        uint ProbeY = ProbeId / ProbeCntPerX;
        uint ProbeLocX; //= ProbeX * (kAyanami_ScreenProbeProbeHemiRes+2);
        uint ProbeLocY; //= ProbeY * (kAyanami_ScreenProbeProbeHemiRes+2);
        ProbeLocX = ProbeX * (kAyanami_ScreenProbeProbeHemiRes+2);
        ProbeLocY = ProbeY * (kAyanami_ScreenProbeProbeHemiRes+2);
        uint2 ProbeLoc = uint2(ProbeLocX, ProbeLocY);
        uint2 WritingSlot = ProbeLoc + TraceRayCoord;
        return WritingSlot; 
    }

    uint ScreenProbePackLocationAndRay(uint ProbeId, uint2 RayId){
        // low->high: (3bit rayId_x, 3bit rayId_y,  26bit probeId)
        uint RayIdXEnc = (RayId.x & 0x7);
        uint RayIdYEnc = (RayId.y & 0x7) << 3;
        uint ProbeXEnc = (ProbeId & 0x3FFFFFF) << 6;
        uint PackedCoord = RayIdXEnc | RayIdYEnc | ProbeXEnc;
        return PackedCoord;
    }

    float4 GetWorldPosFromDepthPersp(PerFramePerViewData PerFrame,float Depth, float2 UV){
        float3 NdcXyz = float3(UV*2.0-1.0, Depth);
        float CamNear = PerFrame.m_CameraNear;
        float CamFar = PerFrame.m_CameraFar;
        float WorldDepth = Math::ClipDepthToViewDepth(Depth, CamNear, CamFar);
        float4 Ndc = float4(NdcXyz, 1.0) * WorldDepth;
        float4x4 ClipToWorld = PerFrame.m_ClipToWorld;
        float4 WorldPos = mul(ClipToWorld, Ndc);
        return float4(WorldPos.xyz / WorldPos.w, WorldDepth);
    }

    uint2 GetRadiosityProbeSHAtlasCoord(uint ProbeIndex,uint CardAtlasResolution, uint CardResolution){

    #if INTERNAL_AYANAMI_RADIOSITY_TILE_SHCONV_DEBUG
        uint ProbesPerTile = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityProbesPerCardTileWidth;
        uint TileIndex = ProbeIndex / ProbesPerTile;
        uint ProbeRemainder = ProbeIndex % ProbesPerTile;
    
        uint TilesPerCardWidth = CardResolution / kAyanami_CardTileWidth;
        uint TilesPerCard = TilesPerCardWidth * TilesPerCardWidth;
        uint CardIndex = TileIndex / TilesPerCard;
    
        uint ReqStoreWidthPerTileWidth = kAyanami_RadiosityProbesPerCardTileWidth;
        uint ReqStoreWidthPerCardWidth = ReqStoreWidthPerTileWidth * TilesPerCardWidth;
    
        uint InTileProbeX = ProbeRemainder % ReqStoreWidthPerTileWidth;
        uint InTileProbeY = ProbeRemainder / ReqStoreWidthPerTileWidth;
        uint2 InTileProbeOffset = uint2(InTileProbeX, InTileProbeY);
    
        uint TileRemainder = TileIndex % TilesPerCard;
        uint TileX = TileRemainder % TilesPerCardWidth;
        uint TileY = TileRemainder / TilesPerCardWidth;
        uint2 TileOffset = uint2(TileX, TileY) * ReqStoreWidthPerTileWidth;
    
        uint CardPerAltasWidth = CardAtlasResolution / CardResolution;
        uint CardX = CardIndex % CardPerAltasWidth;
        uint CardY = CardIndex / CardPerAltasWidth;
        uint2 CardOffset = uint2(CardX, CardY) * ReqStoreWidthPerCardWidth;
    
        // Calculate the final coordinates in the SH atlas
        uint2 FinalOffset = CardOffset + TileOffset + InTileProbeOffset;
        return uint2(FinalOffset.x, FinalOffset.y);
    
    #else
    
        uint TilesPerAtlasWidth = PushConst.m_CardAtlasResolution / kAyanami_CardTileWidth;
        uint ProbesPerAtlasWidth = kAyanami_RadiosityProbesPerCardTileWidth * TilesPerAtlasWidth;
        uint ProbeX = ProbeIndex % ProbesPerAtlasWidth;
        uint ProbeY = ProbeIndex / ProbesPerAtlasWidth;
        return uint2(ProbeX, ProbeY);
    #endif
    }

    void GetRadiosityRayTraceCoordToCardInfo(uint GThreadId, float2 Jitter, out uint2 OffsetInCardTile,out uint CardTileId, out uint2 TraceRayCoord){
        uint ProbeId = GThreadId / kAyanami_RadiosityTracesPerProbe;
        
        uint ProbesPerTile = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityProbesPerCardTileWidth;
        uint CurTileId = ProbeId / ProbesPerTile;
        uint CurProbeIdInTile = ProbeId % ProbesPerTile;
        uint2 ProbePosInTile = uint2(CurProbeIdInTile % kAyanami_RadiosityProbesPerCardTileWidth,
                                     CurProbeIdInTile / kAyanami_RadiosityProbesPerCardTileWidth);
    
        uint ProbeSpacing = kAyanami_CardTileWidth / kAyanami_RadiosityProbesPerCardTileWidth;
        uint2 ProbeOffset = ProbePosInTile * ProbeSpacing + uint2(ProbeSpacing * Jitter);
        OffsetInCardTile = ProbeOffset;
        CardTileId = CurTileId;
    
        uint RayOffsetInProbe =  GThreadId % kAyanami_RadiosityTracesPerProbe;
        uint2 RayCoord = uint2(RayOffsetInProbe % kAyanami_RadiosityProbHemiRes,
                                 RayOffsetInProbe / kAyanami_RadiosityProbHemiRes);
        TraceRayCoord = RayCoord;
    }
}
}