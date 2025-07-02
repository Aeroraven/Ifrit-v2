#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        FDenseGridStructureHandle m_Grid;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridResetCS(uint3 ThreadId:SV_DispatchThreadID,
                        uint3 GroupId:SV_GroupID,
                        uint3 GroupThreadId:SV_GroupThreadID)
    {
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        int TotalGrids = Grid.GetTotalGridCount();
        if(ThreadId.x >= TotalGrids)
            return;
        FSpatialIndex GridIndex = Grid.DecodeSpatialIndex((int)ThreadId.x);

        Grid.StoreForce(GridIndex, FSpatialVector(0.0f));
        Grid.StoreMass(GridIndex, 0.0f);
        Grid.StoreVelocity(GridIndex, FSpatialVector(0.0f));
    }

}}} 