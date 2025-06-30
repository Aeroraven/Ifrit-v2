#include "ifrit.shader.neo/Siro/APIC/APICFluid.Utility.hlsli"

namespace IfritShader{
namespace Siro{

    struct PushConst_T
    {
        float2 m_DefaultGravity;
        uint m_NumGrids;
        uint m_GridSize;        
        float m_DefaultTimestep;
        TRWStructuredBufferHandle<float2> m_GridVelocity;
        TRWStructuredBufferHandle<float> m_GridMass;
        TRWStructuredBufferHandle<int> m_GridState;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void ApicGridUpdateCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if(ThreadId.x >= PushConst.m_NumGrids)
            return;
        int2 GridPos = GetGridPos(ThreadId.x, PushConst.m_GridSize);
        int GridState = PushConst.m_GridState.Load(ThreadId.x);

        if(GridState == 0)
        {
            PushConst.m_GridVelocity.Store(float2(0.0f, 0.0f), ThreadId.x);
            PushConst.m_GridMass.Store(0.0f, ThreadId.x);
        }
        else
        {
            float2 GridMomentum = PushConst.m_GridVelocity.Load(ThreadId.x);
            float GridMass = PushConst.m_GridMass.Load(ThreadId.x);
            float2 GridVelocity = GridMomentum / GridMass;
            //GridVelocity += PushConst.m_DefaultGravity * PushConst.m_DefaultTimestep;

            int BoundaryCondition = 0;
            int NumWeights = 0;
            if(GridPos.x != 0){
                BoundaryCondition |= 1; 
                NumWeights++;
            } 
            if(GridPos.x != PushConst.m_GridSize - 1){
                BoundaryCondition |= 2; 
                NumWeights++;
            }
            if(GridPos.y != 0){
                BoundaryCondition |= 4; 
                NumWeights++;
            }
            if(GridPos.y != PushConst.m_GridSize - 1){
                BoundaryCondition |= 8; 
                NumWeights++;
            }
            NumWeights = NumWeights & 0x7; 
            int PackedState = 1 | (BoundaryCondition << 1) | (NumWeights << 5);

            float2 BoundaryVelocity = float2(0.0f, 0.0f);
            if((BoundaryCondition & 3) == 3)
            {
                BoundaryVelocity.x = GridVelocity.x;
            }
            else if((BoundaryCondition & 12) == 12)
            {
                BoundaryVelocity.y = GridVelocity.y;
            }

            PushConst.m_GridState.Store(PackedState, ThreadId.x);
            PushConst.m_GridVelocity.Store(GridVelocity, ThreadId.x);
        }
    }

}
}
