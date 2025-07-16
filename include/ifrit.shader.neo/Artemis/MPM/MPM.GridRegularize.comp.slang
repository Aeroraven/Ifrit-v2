#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        FDenseGridStructureHandle m_Grid;
        FValidGridCounter m_ValidGridCounter;
        TRWStructuredBufferHandle<int> m_ValidGridList;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

#if !IFSHADER_MPM_WAVE_INTRINSIC_ENABLED
    groupshared int sMarkedGrids[kMpmTGSizeX];
    groupshared int sMarkedGridCount;
    groupshared int sMarkedGridListStart;
#endif

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridRegularizeCS(uint3 ThreadId:SV_DispatchThreadID,
                             uint3 GroupId:SV_GroupID,
                             uint3 GroupThreadId:SV_GroupThreadID)
    {
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        int TotalGrids = Grid.GetTotalGridCount();

#if IFSHADER_MPM_WAVE_INTRINSIC_ENABLED

        FSpatialIndex GridIndex = 0;
        FScalar GridMass = 0.0f;
        FSpatialVector GridMomentum = FSpatialVector(0.0f);
        FSpatialVector GridVelocity = FSpatialVector(0.0f);
        bool IsGridFilled = false;
        if(ThreadId.x < TotalGrids)
        {
            GridIndex = Grid.DecodeSpatialIndex((int)ThreadId.x);
            GridMass = Grid.LoadMass(GridIndex);
            GridMomentum = Grid.LoadVelocity(GridIndex);
            if(GridMass > 1e-6f)
            {
                GridVelocity = GridMomentum / GridMass;
                IsGridFilled = true;
            }
            Grid.StoreVelocity(GridIndex, GridVelocity);
        }

        int sWaveMarkedGridListStart = 0;
        int sThreadMarkedGridListOffset = WavePrefixCountBits(IsGridFilled);
        int sWaveNumGridsToMark = WaveActiveCountBits(IsGridFilled);
        if(WaveIsFirstLane())
        {
            sWaveMarkedGridListStart = PushConst.m_ValidGridCounter.Allocate(sWaveNumGridsToMark);
        }
        int sThreadMarkedGridListStart = WaveReadLaneFirst(sWaveMarkedGridListStart);
        int PositionInList = sThreadMarkedGridListStart + sThreadMarkedGridListOffset;

        if(IsGridFilled)
        {
            int GridId = ThreadId.x;
            PushConst.m_ValidGridList.Store(GridId, PositionInList);
        }

#else
        if(GroupThreadId.x == 0)
            sMarkedGridCount = 0;
        

        GroupMemoryBarrierWithGroupSync();

        if(ThreadId.x < TotalGrids)
        {
            FSpatialIndex GridIndex = Grid.DecodeSpatialIndex((int)ThreadId.x);
            FScalar GridMass = Grid.LoadMass(GridIndex);
            FSpatialVector GridMomentum = Grid.LoadVelocity(GridIndex);
            FSpatialVector GridVelocity = FSpatialVector(0.0f);
            if(GridMass > 1e-6f){
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
#endif

    }

}}}