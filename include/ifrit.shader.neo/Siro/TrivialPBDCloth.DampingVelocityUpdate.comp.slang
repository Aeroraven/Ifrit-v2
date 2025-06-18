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

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Siro/TrivialPBDCloth.Shared.hlsli"

namespace IfritShader {
namespace Siro {
    // Constants for the cloth simulation
    IFSHADER_DEFINE_CONST_UINT32(kBlockSize, 256)

    struct PushConstant_T
    {
        uint m_NumVertices;          // Total number of vertices
        float m_RigidBodyDamping;    // Advanced damping coefficient (0=none, 1=full)
        float m_Epsilon;             // Small epsilon to avoid division by zero
        float m_Pad;                 // Padding to ensure alignment
        
        // Buffer handles
        TRWStructuredBufferHandle<float4> m_PositionBuffer;        // xyz=position, w=inv_mass
        TRWStructuredBufferHandle<float3> m_VelocityBuffer;        // Velocities
        TStructuredBufferHandle<uint> m_FixedVertexBuffer;         // Fixed vertices (1=fixed, 0=movable)
        TStructuredBufferHandle<SystemState> m_SystemStateBuffer;   // Global system state from previous pass
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;
    
    // Compute shader entry point for applying advanced damping to vertices
    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kBlockSize, 1, 1)  // 1D thread blocks for processing vertices
    void TrivialPBDClothDampingVelocityUpdate(
        uint3 ThreadId : SV_DispatchThreadID
    )
    {
        uint index = ThreadId.x;
        
        // Early out if this thread is outside the cloth dimensions
        if (index >= PushConst.m_NumVertices)
            return;
            
        // Read the current vertex position and velocity
        float4 position = PushConst.m_PositionBuffer.Load(index);
        float3 position3 = position.xyz;
        float3 velocity = PushConst.m_VelocityBuffer.Load(index);
        float invMass = position.w;
        
        // Skip fixed vertices or vertices with zero inv_mass (immovable objects)
        uint isFixed = PushConst.m_FixedVertexBuffer.Load(index);
        if (isFixed == 1 || invMass <= 0.0f)
            return;

        // Get system state from the buffer (computed in the previous pass)
        SystemState systemState = PushConst.m_SystemStateBuffer.Load(0);  // Single entry at index 0
        
        // Calculate the relative position vector from the center of mass
        // r_i = x_i - x_cm
        float3 r = position3 - systemState.centerOfMass;
        
        // Calculate the rigid body velocity at this vertex position
        // v_rigid = v_cm + ω × r_i
        float3 angularComponent = cross(systemState.angularVelocity, r);
        float3 rigidVelocity = systemState.centerOfMassVelocity + angularComponent;
        
        // Calculate the velocity correction needed
        // Δv_i = v_rigid - v_i
        float3 velocityDelta = rigidVelocity - velocity;
        
        // Apply damping to the vertex velocity
        // v_i = v_i + k_damping * Δv_i
        velocity += PushConst.m_RigidBodyDamping * velocityDelta;
        
        // Store the updated velocity back to the buffer
        PushConst.m_VelocityBuffer.Store(velocity, index);
    }
    
} // namespace Siro
} // namespace IfritShader