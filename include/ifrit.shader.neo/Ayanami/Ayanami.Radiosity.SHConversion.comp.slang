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
#include "ifrit.shader.neo/Math.SphericalHarmonics.hlsli"

namespace IfritShader{
namespace Ayanami{

    struct PushConstant_T
    {
        float2 m_TraceCoordJitter;
        float2 m_ProbeCenterJitter;
        uint m_CardAtlasResolution;
        uint m_CardResolution;
        uint m_NumTotalCards;
        TTexture2DHandle<float> m_CardDepthAtlasSRV;
        TTexture2DHandle<float4> m_CardNormalAtlasSRV;
        TRWStructuredBufferHandle<CardTransformData> m_AllCardObjData;
        TRWStructuredBufferHandle<FMeshDFDescriptor> m_AllMeshDFData;
        TRWTexture2DHandle<float4> m_FilteredRadianceAtlasUAV;
        TRWTexture2DHandle<float4> m_RWRadiosityProbeSHAtlasRUAV;
        TRWTexture2DHandle<float4> m_RWRadiosityProbeSHAtlasGUAV;
        TRWTexture2DHandle<float4> m_RWRadiosityProbeSHAtlasBUAV;
        uint m_TotalProbes;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    void WriteSHAtlas(uint ProbeIndex, Math::FTwoBandSHRgb SHCoefs){
        uint2 WriteLocation = GetRadiosityProbeSHAtlasCoord(ProbeIndex, PushConst.m_CardAtlasResolution, PushConst.m_CardResolution);
        PushConst.m_RWRadiosityProbeSHAtlasRUAV.Store(WriteLocation, SHCoefs.m_Red.m_Coef);
        PushConst.m_RWRadiosityProbeSHAtlasGUAV.Store(WriteLocation, SHCoefs.m_Green.m_Coef);
        PushConst.m_RWRadiosityProbeSHAtlasBUAV.Store(WriteLocation, SHCoefs.m_Blue.m_Coef);
    }
    

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiSphericalHarmonicsCvtKernelSize,
        1,
        1
    )
    void RadiositySHConversionCS(uint3 ThreadID : SV_DispatchThreadID,
                                 uint3 GroupID : SV_GroupID,
                                 uint3 GroupThreadID : SV_GroupThreadID)
    {
        uint GlobalId = ThreadID.x; //Probe Id
        if(GlobalId >= PushConst.m_TotalProbes) return;
        uint ProbeRayStart = kAyanami_RadiosityTracesPerProbe * GlobalId;

        uint TileIndex;
        uint2 OffsetInTile;
        uint2 TraceRayCoord;

        GetRadiosityRayTraceCoordToCardInfo(ProbeRayStart, PushConst.m_TraceCoordJitter, OffsetInTile, TileIndex, TraceRayCoord);

    }
}
}