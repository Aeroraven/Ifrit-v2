#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        FDenseGridStructureHandle m_Grid;
        FValidGridCounter m_ValidGridCounter;
        TRWStructuredBufferHandle<int> m_ValidGridList;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;
    groupshared int sMarkedGrids[kMpmTGSizeX];
    groupshared int sMarkedGridCount;
    groupshared int sMarkedGridListStart;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridRegularizeCS(uint3 ThreadId:SV_DispatchThreadID,
                             uint3 GroupId:SV_GroupID,
                             uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Step2. compute grid velocities
        // Step3. identify grid degree of freedoms

        if(GroupThreadId.x == 0)
            sMarkedGridCount = 0;
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        int TotalGrids = Grid.GetTotalGridCount();

        GroupMemoryBarrierWithGroupSync();

        if(ThreadId.x < TotalGrids)
        {
            FSpatialIndex GridIndex = Grid.DecodeSpatialIndex((int)ThreadId.x);
            FScalar GridMass = Grid.LoadMass(GridIndex);
            FSpatialVector GridMomentum = Grid.LoadVelocity(GridIndex);
            FSpatialVector GridVelocity = FSpatialVector(0.0f);
            if(GridMass > 0.0f){
                GridVelocity = GridMomentum / GridMass;
                int InGroupLoc;
                InterlockedAdd(sMarkedGridCount, 1, InGroupLoc);
                if(InGroupLoc < kMpmTGSizeX)
                {
                    sMarkedGrids[InGroupLoc] = ThreadId.x;
                }
            }
            Grid.StoreVelocity(GridIndex, GridVelocity);
        }

        GroupMemoryBarrierWithGroupSync();

        if(GroupThreadId.x == 0)
        {
            sMarkedGridListStart = PushConst.m_ValidGridCounter.Allocate(sMarkedGridCount);
        }

        GroupMemoryBarrierWithGroupSync();

        if(GroupThreadId.x < sMarkedGridCount)
        {
            int GridIndex = sMarkedGrids[GroupThreadId.x];
            if(GridIndex >= 0)
            {
                PushConst.m_ValidGridList.Store(GridIndex, sMarkedGridListStart + GroupThreadId.x);
            }
        }

    }

}}}