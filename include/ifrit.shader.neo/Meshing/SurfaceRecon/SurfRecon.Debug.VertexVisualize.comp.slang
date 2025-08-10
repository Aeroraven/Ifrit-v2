#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TStructuredBufferHandle<FSurfReconGrid> m_Grid; //SRV
        TAtomicRWStructuredBufferHandle<int> m_ActiveVerticesCounter;
        TRWStructuredBufferHandle<int> m_ActiveVerticesList;
        TRWStructuredBufferHandle<float> m_VertexDensity;
        TRWTexture2DHandle<float4> m_DebugTex;
        int m_RtW;
        int m_RtH;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSurfReconVertexDensityTGSz, 1, 1)
    void SurfReconDebugVertexVisualizeCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        FSpatialIndex CellsMaxBound = Grid.GetGridNumCellsInAxis()-1;
        int TotalActiveVertices = PushConst.m_ActiveVerticesCounter.Load(0);
        if(ThreadId.x >= TotalActiveVertices) return;
        int VertexIdx = PushConst.m_ActiveVerticesList.Load(ThreadId.x);

        FSpatialIndex VertexCoord = Grid.GetCellVertexFromFlattenId(VertexIdx);
        FSpatialVector VertexLocation = Grid.GetCellVertexPosition(VertexCoord);
        FSpatialIndex MaxBound = min(CellsMaxBound,VertexCoord);
        FSpatialIndex MinBound = max(FSpatialIndex(0),VertexCoord-1);
        float VertexDensity = PushConst.m_VertexDensity.Load(VertexIdx);

        float4 MVPCol1 = float4(0.92376, 0.00, 0.00, 0.00);
        float4 MVPCol2 = float4(0.0f, -1.73205f, 0.0f, 0.0f);
        float4 MVPCol3 = float4(0.0f, 0.0f, 1.0001f, 1.0f);
        float4 MVPCol4 = float4(-0.46188, 0.86603, 0.40004, 0.50);

        float4x4 MVP = transpose(float4x4(MVPCol1, MVPCol2, MVPCol3, MVPCol4));

        float4 ProjectedVertex = mul(MVP, float4(VertexLocation, 1.0f));
        float3 NormalizedVertex = ProjectedVertex.xyz / ProjectedVertex.w;
        float2 UV = (NormalizedVertex.xy + 1.0f) * 0.5f;

        float4 Color = float4(0.0f, 0.0f, 1.0f, 1.0f);
        if(VertexDensity <= 1e-6f)
        {
            Color = float4(1.0f, 0.0f, 0.0f, 1.0f);
        }
        else if(VertexDensity > 0.0f)
        {
            Color = float4(0.0f, 1.0f, 0.0f, 1.0f);
        }
        int U = int(UV.x * PushConst.m_RtW);
        int V = int(UV.y * PushConst.m_RtH);
        uint2 ActualUV = uint2(U, V);
        PushConst.m_DebugTex.Store(ActualUV, Color);
    }

}}}
