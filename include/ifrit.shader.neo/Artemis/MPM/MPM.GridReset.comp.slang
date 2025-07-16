#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#pragma ifrit.multi_compile IFSHADER_MPM_GRIDRESET_INIT
#pragma ifrit.multi_compile IFSHADER_MPM_PBMPM
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        uint m_FirstIter;
        FDenseGridStructureHandle m_Grid;
        FValidGridCounter m_ValidGridCounter;
        TRWStructuredBufferHandle<int> m_ValidGridList;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridResetCS(uint3 ThreadId:SV_DispatchThreadID,
                        uint3 GroupId:SV_GroupID,
                        uint3 GroupThreadId:SV_GroupThreadID)
    {
#ifdef IFSHADER_MPM_GRIDRESET_INIT
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        int TotalGrids = Grid.GetTotalGridCount();
        if(ThreadId.x >= TotalGrids)
            return;
        FSpatialIndex GridIndex = Grid.DecodeSpatialIndex((int)ThreadId.x);
#else
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        int TotalGrids = PushConst.m_ValidGridCounter.GetValidGridCount();
        if(ThreadId.x >= TotalGrids)
            return;
        FSpatialIndex GridIndex =  Grid.DecodeSpatialIndex(PushConst.m_ValidGridList.Load(ThreadId.x));
#endif

#ifndef IFSHADER_MPM_PBMPM
         Grid.StoreForce(GridIndex, FSpatialVector(0.0f));
         Grid.StoreMass(GridIndex, 0.0f);
         Grid.StoreVelocity(GridIndex, FSpatialVector(0.0f));
#else
        bool IsFirstIter = (PushConst.m_FirstIter & 1) != 0;
        if(IsFirstIter)
        {
            Grid.StoreMass(GridIndex, 0.0f);
        }
        Grid.StoreVelocity(GridIndex, FSpatialVector(0.0f));
#endif
    }

}}} 