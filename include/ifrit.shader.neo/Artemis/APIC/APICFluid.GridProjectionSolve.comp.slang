#include "ifrit.shader.neo/Artemis/APIC/APICFluid.Utility.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T
    {
        uint m_GridSize;
        TRWStructuredBufferHandle<float> m_GridCoefB; // Ax=B
        TRWStructuredBufferHandle<int> m_GridState;
        TRWStructuredBufferHandle<float> m_GridLastPressure;
        TRWStructuredBufferHandle<float> m_GridNewPressure;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    float GetLastPressure(int2 GridPos, uint GridSize)
    {
        int GridIndex = GetGridIndex(GridPos, GridSize);
        return PushConst.m_GridLastPressure.Load(GridIndex);
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kArtemisTGSizeX,1,1)
    void ApicGridProjectionSolveCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        // Obtain a divergence-free velocity field u*
        // du/dt = -1/rho*Grad(p), s.t. Div(u*)=0
        // => u* = u - dt/rho*Grad(p)
        // => Div(u*) = Div(u) - dt/rho*Laplacian(p)
        // => Div(u) = K*Laplacian(p)
        // Div(u) = du_x/dx + du_y/dy = (Ux_r-Ux_l)/(2*dx)+(Uy_b-Uy_t)/(2*dx)
        // Laplacian(p) = (p_r + p_l + p_b + p_t - 4*p_c)/(dx*dx) 
        // 4*pc-pr-pl-pb-pt = K*(ux_l-ux_r)/2 + K*(uy_t-uy_b)/(2)

        uint TotalGrids = PushConst.m_GridSize * PushConst.m_GridSize;
        if(ThreadId.x >= TotalGrids)
            return;

        int GridState = PushConst.m_GridState.Load(ThreadId.x);
        if( (GridState & 0x1) == 0)
        {
            PushConst.m_GridNewPressure.Store(0.0f, ThreadId.x);
        }
        else
        {
            float B = -PushConst.m_GridCoefB.Load(ThreadId.x);
            int WeightCentersI = (GridState >> 5) & 0x7;
            float WeightCenters = (float)WeightCentersI;
            int BoundaryCondition = (GridState >> 1) & 0xF;
            int2 GridPos = GetGridPos((int)ThreadId.x, PushConst.m_GridSize);

            float Sum = 0.0f;
            if((BoundaryCondition & 0x1) == 0x1)
            {
                Sum -= GetLastPressure(GridPos + int2(-1, 0), PushConst.m_GridSize);
            }
            if((BoundaryCondition & 0x2) == 0x2)
            {
                Sum -= GetLastPressure(GridPos + int2(1, 0), PushConst.m_GridSize);
            }
            if((BoundaryCondition & 0x4) == 0x4)
            {
                Sum -= GetLastPressure(GridPos + int2(0, -1), PushConst.m_GridSize);
            }
            if((BoundaryCondition & 0x8) == 0x8)
            {
                Sum -= GetLastPressure(GridPos + int2(0, 1), PushConst.m_GridSize);
            }

            float CenterPressure = GetLastPressure(GridPos, PushConst.m_GridSize);
            float Residual = (B - Sum);
            float NewPressure = Residual / WeightCenters;

            float MixedPressure = lerp(CenterPressure, NewPressure, 0.5f);
            PushConst.m_GridNewPressure.Store(NewPressure, ThreadId.x);
        }

    }

}
}
