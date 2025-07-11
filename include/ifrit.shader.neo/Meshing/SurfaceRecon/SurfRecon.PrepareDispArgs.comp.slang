#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TStructuredBufferHandle<int> m_ParticleCounterSrc; 
        TRWStructuredBufferHandle<int> m_ParticleCounterDst; //(cnt,dispx,1,1)
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void SurfReconPrepareDispArgsCS(uint3 ThreadId:SV_DispatchThreadID,
                                    uint3 GroupId:SV_GroupID,
                                    uint3 GroupThreadId:SV_GroupThreadID)
    {
        if(ThreadId.x == 0)
        {
            int NumParticles = PushConst.m_ParticleCounterSrc.Load(0);
            PushConst.m_ParticleCounterDst.Store(NumParticles,0);
            PushConst.m_ParticleCounterDst.Store(DivRoundUp(NumParticles,kSurfReconGridBuildTGSz),1);
            PushConst.m_ParticleCounterDst.Store(1,2);
            PushConst.m_ParticleCounterDst.Store(1,3);
        }
    }

}}}