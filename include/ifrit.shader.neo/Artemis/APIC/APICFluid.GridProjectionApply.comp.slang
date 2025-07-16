#include "ifrit.shader.neo/Artemis/APIC/APICFluid.Utility.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T
    {
        uint m_GridSize;
        float m_Timestep;
        float m_Density;
        TRWStructuredBufferHandle<float2> m_GridVelocity;
        TRWStructuredBufferHandle<float> m_GridPressure;
        TRWStructuredBufferHandle<int> m_GridState;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    float GetPressure(int2 GridPos, uint GridSize)
    {
        int GridIndex = GetGridIndex(GridPos, GridSize);
        return PushConst.m_GridPressure.Load(GridIndex);
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kArtemisTGSizeX,1,1)
    void ApicGridProjectionApplyCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        // => u* = u - dt/rho*Grad(p)
        uint TotalGrids = PushConst.m_GridSize * PushConst.m_GridSize;
        float K = PushConst.m_Timestep / PushConst.m_Density;
        if(ThreadId.x >= TotalGrids)
            return;

        int GridState = PushConst.m_GridState.Load(ThreadId.x);
        int BoundaryCondition = (GridState >> 1) & 0xF;
        int2 GridPos = GetGridPos((int)ThreadId.x, PushConst.m_GridSize);

        float2 GradPressure = float2(0.0f, 0.0f);

        if((BoundaryCondition & 0x3) == 0x3)
        {
            float PressureDxR = GetPressure(GridPos + int2(1, 0), PushConst.m_GridSize);
            float PressureDxL = GetPressure(GridPos + int2(-1, 0), PushConst.m_GridSize);
            float PressureDx = (PressureDxR - PressureDxL) * 0.5f;
            GradPressure.x = PressureDx;
        }else if((BoundaryCondition & 0x1) == 0x1)
        {
            float PressureDxL = GetPressure(GridPos + int2(-1, 0), PushConst.m_GridSize);
            float PressureDxR = GetPressure(GridPos, PushConst.m_GridSize);
            float PressureDx = PressureDxR-PressureDxL;
            GradPressure.x = PressureDx;
        }
        else if((BoundaryCondition & 0x2) == 0x2)
        {
            float PressureDxR = GetPressure(GridPos + int2(1, 0), PushConst.m_GridSize);
            float PressureDxL = GetPressure(GridPos, PushConst.m_GridSize);
            float PressureDx = PressureDxR - PressureDxL;
            GradPressure.x = PressureDx;
        }

        if((BoundaryCondition & 0xC) == 0xC)
        {
            float PressureDyB = GetPressure(GridPos + int2(0, 1), PushConst.m_GridSize);
            float PressureDyT = GetPressure(GridPos + int2(0, -1), PushConst.m_GridSize);
            float PressureDy = (PressureDyB - PressureDyT) * 0.5f;
            GradPressure.y = PressureDy;
        }
        else if((BoundaryCondition & 0x4) == 0x4)
        {
            float PressureDyT = GetPressure(GridPos + int2(0, -1), PushConst.m_GridSize);
            float PressureDyB = GetPressure(GridPos, PushConst.m_GridSize);
            float PressureDy = PressureDyB - PressureDyT;
            GradPressure.y = PressureDy;
        }
        else if((BoundaryCondition & 0x8) == 0x8)
        {
            float PressureDyB = GetPressure(GridPos + int2(0, 1), PushConst.m_GridSize);
            float PressureDyT = GetPressure(GridPos, PushConst.m_GridSize);
            float PressureDy = PressureDyB - PressureDyT;
            GradPressure.y = PressureDy;
        }
    
        float2 GridVelocity = PushConst.m_GridVelocity.Load(ThreadId.x);
        GridVelocity -= K * GradPressure;

        PushConst.m_GridVelocity.Store(GridVelocity, ThreadId.x);
    }

}
}
