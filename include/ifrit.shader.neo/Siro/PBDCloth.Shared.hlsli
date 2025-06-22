/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#ifndef __cplusplus
    #include "ifrit.shader.neo/Bindless.hlsli"
    #include "ifrit.shader.neo/ObjectDescriptor.hlsli"
#endif // __cplusplus

namespace IfritShader{
namespace Siro{

    IFSHADER_DEFINE_CONST_UINT32(kSiroTGSizeX, 128)

#ifndef __cplusplus
    struct FPBDCorrectionHandle
    {
        TAtomicRWStructuredBufferHandle<float> m_Data;

        void AddCorrection(uint Index, float3 Correction)
        {
            uint Offset = Index * 4;
            m_Data.AtomicAdd(Offset, Correction.x);
            m_Data.AtomicAdd(Offset + 1, Correction.y);
            m_Data.AtomicAdd(Offset + 2, Correction.z);
        }
    };

    struct FPBDDistanceConstraint
    {
        uint m_ParticleA;
        uint m_ParticleB;
        float m_RestLength;
        float m_Stiffness;
    };

    struct FPBDBendingConstraint
    {
        uint m_ParticleA;
        uint m_ParticleB;
        uint m_ParticleC;
        uint m_ParticleD;
        float m_RestAngle;
        float m_Stiffness;
    }
#endif // __cplusplus
}
}