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

#include "ifrit.shader.neo/Siro/PBDCloth.Shared.hlsli"
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"
#include "ifrit.shader.neo/ObjectDescriptor.hlsli"

namespace IfritShader{
namespace Siro{

    struct PushConst_T{
        FPBDCollisionProcessIndirectArgs m_CollisionConstraintCounter;
        TRWStructuredBufferHandle<Ayanami::FMeshDFDetail> m_CollisionSDFs;
        TRWStructuredBufferHandle<FPerObjectData> m_ObjectData;
        TRWStructuredBufferHandle<FPBDCollsionConstraint> m_CollisionConstraints;
        uint m_NumParticles;
        uint m_NumSDFs;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothGenerateSDFCollisionCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_NumParticles) return;
        

    }
}
}