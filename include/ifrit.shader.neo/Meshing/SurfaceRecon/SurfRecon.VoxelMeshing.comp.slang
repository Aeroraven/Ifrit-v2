#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"
#include "ifrit.shader.neo/Meshing/MarchingCubesLUT.hlsli"

// NO mesh shader support for now x_x

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TStructuredBufferHandle<int> m_ActiveGridBlockList;
        TStructuredBufferHandle<FSurfReconGrid> m_Grid; 
        TRWStructuredBufferHandle<int> m_IndexBuffer;
        TRWStructuredBufferHandle<float4> m_VertexBuffer; 
        TRWStructuredBufferHandle<float4> m_NormalBuffer;
        TRWStructuredBufferHandle<float4> m_TangentBuffer;
        TRWStructuredBufferHandle<float2> m_UVBuffer;

        TAtomicRWStructuredBufferHandle<int> m_TriangleCounter;

        TStructuredBufferHandle<float> m_VertexDensity;
        FSpatialVectorHandle m_CellVertexNormal; //SRV
        float m_IsoValue;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    float InterpolateDensity(float DensityS, float DensityE)
    {
        return clamp((PushConst.m_IsoValue-DensityS)/(DensityE-DensityS), 0.0f, 1.0f);
    }

    int LoadCellAsMCMask(FSpatialIndex CellCoord)
    {
        int Mask = 0;
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        for(int i = 0; i < 8; ++i)
        {
            FSpatialIndex VertCoord = CellCoord + kMarchingCubesVertLUT[i];
            bool IsMarked = PushConst.m_VertexDensity.Load(Grid.GetCellVertexFlattenId(VertCoord)) > PushConst.m_IsoValue;
            if(IsMarked)
            {
                Mask |= (1 << i);
            }
        }
        return Mask;
    }

    void AppendFace(FSpatialIndex CellCoord, int Link0, int Link1, int Link2)
    {
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        FSpatialIndex NumCells = ToSpatialIndex(Grid.m_NumCells);
        FSpatialVector NumCellsF = FSpatialVector(NumCells);
        FSpatialVector MinLoc = ToSpatialVector(Grid.m_MinBound);
        FSpatialVector MaxLoc = ToSpatialVector(Grid.m_MaxBound);

        int Link0A = kMarchingCubesLinkLUT[Link0][0];
        int Link0B = kMarchingCubesLinkLUT[Link0][1];
        int Link1A = kMarchingCubesLinkLUT[Link1][0];
        int Link1B = kMarchingCubesLinkLUT[Link1][1];
        int Link2A = kMarchingCubesLinkLUT[Link2][0];
        int Link2B = kMarchingCubesLinkLUT[Link2][1];

        FSpatialIndex VertCellIdx0A = CellCoord + kMarchingCubesVertLUT[Link0A];
        FSpatialIndex VertCellIdx0B = CellCoord + kMarchingCubesVertLUT[Link0B];
        FSpatialIndex VertCellIdx1A = CellCoord + kMarchingCubesVertLUT[Link1A];
        FSpatialIndex VertCellIdx1B = CellCoord + kMarchingCubesVertLUT[Link1B];
        FSpatialIndex VertCellIdx2A = CellCoord + kMarchingCubesVertLUT[Link2A];
        FSpatialIndex VertCellIdx2B = CellCoord + kMarchingCubesVertLUT[Link2B];

        float Density0A = PushConst.m_VertexDensity.Load(Grid.GetCellVertexFlattenId(VertCellIdx0A));
        float Density0B = PushConst.m_VertexDensity.Load(Grid.GetCellVertexFlattenId(VertCellIdx0B));
        float Density1A = PushConst.m_VertexDensity.Load(Grid.GetCellVertexFlattenId(VertCellIdx1A));
        float Density1B = PushConst.m_VertexDensity.Load(Grid.GetCellVertexFlattenId(VertCellIdx1B));
        float Density2A = PushConst.m_VertexDensity.Load(Grid.GetCellVertexFlattenId(VertCellIdx2A));
        float Density2B = PushConst.m_VertexDensity.Load(Grid.GetCellVertexFlattenId(VertCellIdx2B));
        float Alpha0 = InterpolateDensity(Density0A, Density0B);
        float Alpha1 = InterpolateDensity(Density1A, Density1B);
        float Alpha2 = InterpolateDensity(Density2A, Density2B);


        FSpatialVector VertOff0A = kMarchingCubesVertLUT[Link0A];
        FSpatialVector VertOff0B = kMarchingCubesVertLUT[Link0B];
        FSpatialVector VertOff1A = kMarchingCubesVertLUT[Link1A];
        FSpatialVector VertOff1B = kMarchingCubesVertLUT[Link1B];
        FSpatialVector VertOff2A = kMarchingCubesVertLUT[Link2A];
        FSpatialVector VertOff2B = kMarchingCubesVertLUT[Link2B];   

        FSpatialVector VertOff0 = lerp(VertOff0A, VertOff0B, Alpha0);
        FSpatialVector VertOff1 = lerp(VertOff1A, VertOff1B, Alpha1);
        FSpatialVector VertOff2 = lerp(VertOff2A, VertOff2B, Alpha2);

        FSpatialVector Norm0A = PushConst.m_CellVertexNormal.Load(Grid.GetCellVertexFlattenId(VertCellIdx0A));
        FSpatialVector Norm0B = PushConst.m_CellVertexNormal.Load(Grid.GetCellVertexFlattenId(VertCellIdx0B));
        FSpatialVector Norm1A = PushConst.m_CellVertexNormal.Load(Grid.GetCellVertexFlattenId(VertCellIdx1A));
        FSpatialVector Norm1B = PushConst.m_CellVertexNormal.Load(Grid.GetCellVertexFlattenId(VertCellIdx1B));
        FSpatialVector Norm2A = PushConst.m_CellVertexNormal.Load(Grid.GetCellVertexFlattenId(VertCellIdx2A));
        FSpatialVector Norm2B = PushConst.m_CellVertexNormal.Load(Grid.GetCellVertexFlattenId(VertCellIdx2B));

        FSpatialVector Norm0 = lerp(Norm0A, Norm0B, Alpha0);
        FSpatialVector Norm1 = lerp(Norm1A, Norm1B, Alpha1);
        FSpatialVector Norm2 = lerp(Norm2A, Norm2B, Alpha2);
        

        FSpatialVector VertBase = FSpatialVector(CellCoord);
        FSpatialVector Vert0 = VertBase + VertOff0;
        FSpatialVector Vert1 = VertBase + VertOff1;
        FSpatialVector Vert2 = VertBase + VertOff2;

        Vert0 = Vert0 / NumCellsF;
        Vert1 = Vert1 / NumCellsF;
        Vert2 = Vert2 / NumCellsF;

        Vert0 = lerp(MinLoc, MaxLoc, Vert0);
        Vert1 = lerp(MinLoc, MaxLoc, Vert1);
        Vert2 = lerp(MinLoc, MaxLoc, Vert2);

        int Slot = PushConst.m_TriangleCounter.AtomicAdd(0, 3);
        PushConst.m_IndexBuffer.Store(Slot, Slot);
        PushConst.m_IndexBuffer.Store(Slot + 1, Slot + 1);
        PushConst.m_IndexBuffer.Store(Slot + 2, Slot + 2);

        PushConst.m_VertexBuffer.Store(float4(Vert0, (float)1.0f),Slot);
        PushConst.m_VertexBuffer.Store(float4(Vert1, (float)1.0f),Slot+1);
        PushConst.m_VertexBuffer.Store(float4(Vert2, (float)1.0f),Slot+2);

        float4 Normal = normalize(float4(1.0f,0.0f,0.0f,0.0f));
        PushConst.m_NormalBuffer.Store(float4(Norm0,0.0f), Slot);
        PushConst.m_NormalBuffer.Store(float4(Norm1,0.0f), Slot + 1);
        PushConst.m_NormalBuffer.Store(float4(Norm2,0.0f), Slot + 2);
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSurfReconFilterCellTGSz3D, kSurfReconFilterCellTGSz3D, kSurfReconFilterCellTGSz3D)
    void SurfReconVoxelMeshingCS(uint3 ThreadId:SV_DispatchThreadID,
                                 uint3 GroupId:SV_GroupID,
                                 uint3 GroupThreadId:SV_GroupThreadID)
    {
        FSpatialIndex InBlockCellOffset = ToSpatialIndex(GroupThreadId.x, GroupThreadId.y, GroupThreadId.z);
        int BlockCandidateId = GroupId.x;
        int BlockId = PushConst.m_ActiveGridBlockList.Load(BlockCandidateId);
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        FSpatialIndex BlockCoord = Grid.GetBlockCoordFromFlattenId(BlockId);
        FSpatialIndex CellCoord = BlockCoord * Grid.m_BlockWidthInCellUnits + InBlockCellOffset;
        
        int MCMask = LoadCellAsMCMask(CellCoord);
        FMarchingCubeLUT MCFaceData = DecodeMarchingCubeLUT(MCMask);

        for(int i = 0; i < MCFaceData.m_NumTriangles; ++i)
        {
            int Link0 = MCFaceData.m_TriIndices[i * 3 + 0];
            int Link1 = MCFaceData.m_TriIndices[i * 3 + 1];
            int Link2 = MCFaceData.m_TriIndices[i * 3 + 2];
            AppendFace(CellCoord, Link0, Link1, Link2);
            
        }
    }

}}}
