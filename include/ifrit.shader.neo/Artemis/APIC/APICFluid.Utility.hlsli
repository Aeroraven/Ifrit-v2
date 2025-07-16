#pragma once
#include "ifrit.shader.neo/Common.hlsli"

#ifndef __cplusplus
    #include "ifrit.shader.neo/Bindless.hlsli"
#endif

namespace IfritShader{
namespace Artemis {

    IFSHADER_DEFINE_CONST_UINT32(kArtemisTGSizeX, 128);

#ifndef __cplusplus
    int GetGridIndex(int2 GridPos, uint GridSize)
    {
        return GridPos.x + GridPos.y * (int)GridSize;
    }

    int2 GetGridPos(int Index, uint GridSize)
    {
        return int2(Index % int(GridSize), Index / int(GridSize));
    }

    
    bool IsValidGridPos(int2 GridPos, uint GridSize)
    {
        return GridPos.x >= 0 && GridPos.x < int(GridSize) &&
               GridPos.y >= 0 && GridPos.y < int(GridSize);
    }

    struct FApicGridVelocityHandle
    {
        uint m_GridSize;
        TAtomicRWStructuredBufferHandle<float> m_GridVelocity;

        void AddVelocity(int2 GridPos, float2 Velocity)
        {
            uint GridIndex = GetGridIndex(GridPos, m_GridSize);
            m_GridVelocity.AtomicAdd(GridIndex * 2, Velocity.x);
            m_GridVelocity.AtomicAdd(GridIndex * 2 + 1, Velocity.y);
        }

        static FApicGridVelocityHandle Create(uint GridSize, TAtomicRWStructuredBufferHandle<float> GridVelocity)
        {
            FApicGridVelocityHandle Handle;
            Handle.m_GridSize = GridSize;
            Handle.m_GridVelocity = GridVelocity;
            return Handle;
        }
    };

    struct FApicGridMassHandle
    {
        uint m_GridSize;
        TAtomicRWStructuredBufferHandle<float> m_GridMass;

        void AddMass(int2 GridPos, float Mass)
        {
            uint GridIndex = GetGridIndex(GridPos, m_GridSize);
            m_GridMass.AtomicAdd(GridIndex, Mass);
        }

        static FApicGridMassHandle Create(uint GridSize, TAtomicRWStructuredBufferHandle<float> GridMass)
        {
            FApicGridMassHandle Handle;
            Handle.m_GridSize = GridSize;
            Handle.m_GridMass = GridMass;
            return Handle;
        }
    };

    float QuadraticInterpWeight(float x)
    {
        float absX = abs(x);
        if (absX < 0.5f)
        {
            return 0.75f - absX * absX;
        }
        else if (absX < 1.5f)
        {
            return 0.5f * (1.5f - absX) * (1.5f - absX);
        }
        return 0.0f;
    }
#endif

}
}