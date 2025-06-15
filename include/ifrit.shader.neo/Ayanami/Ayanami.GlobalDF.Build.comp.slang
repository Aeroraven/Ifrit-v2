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
        uint4 m_GlobalDFTileRange;
        float4 m_GlobalDFWorldRangeMin;
        float4 m_GlobalDFWorldRangeMax;
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        TRWTexture3DHandle<float> m_GlobalDFVolume;
        uint m_NumMeshDF;
        TRWStructuredBufferHandle<FMeshDFDescriptor> m_MeshDFDesc;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiGlobalDFCompositeTileSize,
        kAyanamiGlobalDFCompositeTileSize,
        kAyanamiGlobalDFCompositeTileSize
    )
    void GlobalDFBuildCS(uint3 ThreadID : SV_DispatchThreadID,
                         uint3 GroupID : SV_GroupID,
                         uint3 GroupThreadID : SV_GroupThreadID)
    {
        if(ThreadID.x>=PushConst.m_GlobalDFTileRange.x || 
            ThreadID.y>=PushConst.m_GlobalDFTileRange.y || 
            ThreadID.z>=PushConst.m_GlobalDFTileRange.z) 
            return;
        
        float3 CellNormalizedCoord = (float3(ThreadID)+0.5)/float3(PushConst.m_GlobalDFTileRange.xyz);
        float3 CellWorldCoord = lerp(PushConst.m_GlobalDFWorldRangeMin.xyz, PushConst.m_GlobalDFWorldRangeMax.xyz, CellNormalizedCoord);

        uint TotalMeshes = PushConst.m_NumMeshDF;
        float OptimalSDF = 1e10;

        for(uint CurMesh=0; CurMesh<TotalMeshes; CurMesh++){
            FMeshDFDescriptor CurMeshDesc0 =  PushConst.m_MeshDFDesc.Load(CurMesh);

            float4x4 WorldToLocal = CurMeshDesc0.m_Transform.Load().m_WorldToLocal;
            float3 MeshScale = CurMeshDesc0.m_Transform.Load().m_MaxScale.xyz;
            float MeshMaxScale = max(max(MeshScale.x, MeshScale.y), MeshScale.z);
    
            FMeshDFDetail MdfMeta = CurMeshDesc0.m_MdfDetail.Load();
            float2 MeshDFQuantScale = MdfMeta.GetSdfQuantScale();
    
            float4 CellLocalCoord = mul(WorldToLocal,float4(CellWorldCoord, 1.0));
            float3 CellLocalCoord3 = CellLocalCoord.xyz/CellLocalCoord.w;
    
            float3 MeshBBoxMin = MdfMeta.m_BBoxMin.xyz;
            float3 MeshBBoxMax = MdfMeta.m_BBoxMax.xyz;
            float3 MeshBBoxCenter = (MeshBBoxMin+MeshBBoxMax)*0.5;
            float3 MeshBBoxExtent = (MeshBBoxMax-MeshBBoxMin)*0.5;
    
            float3 CellLocalCoord3RelativeToCenter = CellLocalCoord3-MeshBBoxCenter;
            float3 ToBox = (abs(CellLocalCoord3RelativeToCenter)-MeshBBoxExtent) * MeshScale;
    
            float ToBoxOut = length(max(ToBox, float3(0.0)));
            float ToBoxIn = min(max(ToBox.x,max(ToBox.y,ToBox.z)), 0.0);
            float ToBoxAll = ToBoxOut + ToBoxIn;
            float ToBoxAllPositive = max(ToBoxAll, 0.0);
    
    
            float3 ClampedPos = clamp(CellLocalCoord3RelativeToCenter, -MeshBBoxExtent, MeshBBoxExtent);
            ClampedPos = ClampedPos + MeshBBoxCenter;
    
            float3 ClampedUVW = (ClampedPos-MeshBBoxMin)/(MeshBBoxMax-MeshBBoxMin);
    
            float SdfVal = MdfMeta.SampleSdf(ClampedUVW) * MeshMaxScale;
            float TotalSdf = max(SdfVal + ToBoxAllPositive,ToBoxAll);
    
            OptimalSDF = min(OptimalSDF, TotalSdf);
        }

        PushConst.m_GlobalDFVolume.Store(ThreadID, OptimalSDF);

    }


}
}