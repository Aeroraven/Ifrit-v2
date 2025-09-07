#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TRWStructuredBufferHandle<int> m_DispArgs;
        int m_ThreadBlockSizeX;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void SurfReconComputeDispArgsCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        int TotalElements = PushConst.m_DispArgs.Load(0);
        int NumTGX = DivRoundUp(TotalElements, PushConst.m_ThreadBlockSizeX);
        PushConst.m_DispArgs.Store(NumTGX,1);
        PushConst.m_DispArgs.Store(1,2);
        PushConst.m_DispArgs.Store(1,3);
    }

}}}