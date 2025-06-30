#include "ifrit.shader.neo/Siro/APIC/APICFluid.Utility.hlsli"

namespace IfritShader{
namespace Siro{

    struct PushConst_T
    {
        uint m_GridSize;
        TRWStructuredBufferHandle<float2> m_GridVelocity;
        TRWStructuredBufferHandle<int> m_GridState;
        TRWStructuredBufferHandle<float> m_GridMass;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void ApicGridResetCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        uint TotalGrids = PushConst.m_GridSize * PushConst.m_GridSize;
        if(ThreadId.x >= TotalGrids)
            return;

        PushConst.m_GridVelocity.Store(float2(0.0f, 0.0f), ThreadId.x);
        PushConst.m_GridMass.Store(0.0f, ThreadId.x);
        PushConst.m_GridState.Store(0, ThreadId.x); 
    }

}
}