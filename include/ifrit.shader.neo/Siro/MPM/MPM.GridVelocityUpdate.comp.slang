#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        FScalar m_DeltaTime;
        FDenseGridStructureHandle m_Grid;
        FValidGridCounter m_ValidGridCounter;
        TRWStructuredBufferHandle<int> m_ValidGridList;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridVelocityUpdateCS(uint3 ThreadId:SV_DispatchThreadID,
                                 uint3 GroupId:SV_GroupID,
                                 uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Step5. grid velocity update
        int TotalValidGrids = PushConst.m_ValidGridCounter.GetValidGridCount();
        if(ThreadId.x >= TotalValidGrids)
            return;

        int ValidGridIndex = PushConst.m_ValidGridList.Load(ThreadId.x);
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FSpatialIndex GridCell = Grid.DecodeSpatialIndex(ValidGridIndex);
        FSpatialCondBitSet BoundaryCondition = Grid.IsBoundaryGrid(GridCell);

        FSpatialCondBitSet LeftBoundary = BoundaryCondition & FSpatialCondBitSet(1);
        FSpatialCondBitSet RightBoundary = (BoundaryCondition & FSpatialCondBitSet(2)) >> 1;

        FSpatialVector GridVelocity = Grid.LoadVelocity(GridCell);
        FSpatialVector GridForce = Grid.LoadForce(GridCell);
        FScalar GridMass = Grid.LoadMass(GridCell);

        FSpatialVector NewVelocity = GridVelocity + PushConst.m_DeltaTime * GridForce / GridMass;

        FSpatialVector NewVelocityPos = max(NewVelocity, FSpatialVector(0.0f));
        FSpatialVector NewVelocityNeg = min(NewVelocity, FSpatialVector(0.0f));
        FSpatialVector LeftBoundaryReject = 1.0f - FSpatialVector(LeftBoundary);
        FSpatialVector RightBoundaryReject = 1.0f - FSpatialVector(RightBoundary);

        NewVelocity = NewVelocityPos * RightBoundaryReject + NewVelocityNeg * LeftBoundaryReject;

        Grid.StoreVelocity(GridCell, NewVelocity);
    }

}}}