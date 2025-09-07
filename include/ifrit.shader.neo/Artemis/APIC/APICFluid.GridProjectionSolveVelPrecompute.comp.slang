#include "ifrit.shader.neo/Artemis/APIC/APICFluid.Utility.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T
    {
        float m_Timestep;
        float m_Density;
        uint m_GridSize;
        TRWStructuredBufferHandle<float2> m_GridVelocity;
        TRWStructuredBufferHandle<int> m_GridState;
        TRWStructuredBufferHandle<float> m_GridCoefB; // Ax=B
        TRWStructuredBufferHandle<float> m_GridInitPressure;
        uint m_ClearPressure;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    float2 GetGridVelocity(int2 GridPos)
    {
        int GridIndex = GetGridIndex(GridPos, PushConst.m_GridSize);
        return PushConst.m_GridVelocity.Load(GridIndex);
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kArtemisTGSizeX,1,1)
    void ApicGridProjectionSolveVelPrecomputeCS(uint3 ThreadId:SV_DispatchThreadID)
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
        float K = PushConst.m_Density/PushConst.m_Timestep;
        if(ThreadId.x >= TotalGrids)
            return;

        int GridState = PushConst.m_GridState.Load(ThreadId.x);

        if(PushConst.m_ClearPressure == 1)
        {
            PushConst.m_GridInitPressure.Store(0.5f, ThreadId.x);
        }
        

        if( (GridState & 0x1) == 0)
        {
            PushConst.m_GridCoefB.Store(0.0f, ThreadId.x);
        }
        else
        {
            int WeightCenters = (GridState >> 5) & 0x7;
            int BoundaryCondition = (GridState >> 1) & 0xF;
            int2 GridPos = GetGridPos((int)ThreadId.x, PushConst.m_GridSize);

            // Velocity X
            float VelocityDx = 0.0f;
            if((BoundaryCondition & 0x1) == 0) // left
            {
                VelocityDx = (GetGridVelocity(GridPos + int2(1, 0)) - GetGridVelocity(GridPos)).x;
            }
            else if((BoundaryCondition & 0x2) == 0) // right
            {
                VelocityDx = (GetGridVelocity(GridPos) - GetGridVelocity(GridPos + int2(-1, 0))).x;
            }
            else
            {
                VelocityDx = (GetGridVelocity(GridPos + int2(1, 0)) - GetGridVelocity(GridPos + int2(-1, 0))).x;
                VelocityDx *= 0.5f;
            }

            // Velocity Y
            float VelocityDy = 0.0f;
            if((BoundaryCondition & 0x4) == 0) // top
            {
                VelocityDy = (GetGridVelocity(GridPos + int2(0, 1)) - GetGridVelocity(GridPos)).y;
            }
            else if((BoundaryCondition & 0x8) == 0) // bottom
            {
                VelocityDy = (GetGridVelocity(GridPos) - GetGridVelocity(GridPos + int2(0, -1))).y;
            }
            else
            {
                VelocityDy = (GetGridVelocity(GridPos + int2(0, 1)) - GetGridVelocity(GridPos + int2(0, -1))).y;
                VelocityDy *= 0.5f;
            }

            float DivU = VelocityDx + VelocityDy;
            PushConst.m_GridCoefB.Store(DivU * K, ThreadId.x);
        }

    }

}
}
