#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS

#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        FDenseBlockStructureHandle m_Block;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void MpmParticleScatterPrepareCS(uint3 ThreadId:SV_DispatchThreadID,
                                    uint3 GroupId:SV_GroupID,
                                    uint3 GroupThreadId:SV_GroupThreadID)
    {
        FDenseBlockStructure Block = PushConst.m_Block.Load();
        Block.ClearPages();
    }

}}}
