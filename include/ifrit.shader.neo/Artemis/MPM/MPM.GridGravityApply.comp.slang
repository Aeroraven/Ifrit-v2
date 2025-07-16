#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        float4 m_Gravity;
        FDenseGridStructureHandle m_Grid;
        FValidGridCounter m_ValidGridCounter;
        TRWStructuredBufferHandle<int> m_ValidGridList;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridGravityApplyCS(uint3 ThreadId:SV_DispatchThreadID,
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
        
        FSpatialVector GridForce = Grid.LoadForce(GridCell);
        FScalar GridMass = Grid.LoadMass(GridCell);
        FSpatialVector GravityAcceleration = ToSpatialVector(PushConst.m_Gravity);

        FSpatialVector NewForce = GridForce + GridMass * GravityAcceleration;

        Grid.StoreForce(GridCell, NewForce);
    }

}}}