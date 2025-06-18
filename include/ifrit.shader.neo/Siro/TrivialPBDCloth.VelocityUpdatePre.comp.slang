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

namespace IfritShader {
namespace Siro {
    // Constants for the cloth simulation
    IFSHADER_DEFINE_CONST_UINT32(kBlockSize, 16)
      struct PushConstant_T
    {
        float m_DeltaTime;           // Time step size
        float m_InvMass;             // Inverse mass (1/mass)
        float m_Gravity;             // Gravity force magnitude (y-axis)
        uint m_NumVertices;          // Total number of vertices
        
        // Buffer handles
        TRWStructuredBufferHandle<float4> m_PositionBuffer;        // xyz=position, w=inv_mass
        TRWStructuredBufferHandle<float4> m_PrevPositionBuffer;    // Previous positions
        TRWStructuredBufferHandle<float3> m_VelocityBuffer;        // Velocities
        TStructuredBufferHandle<uint> m_FixedVertexBuffer;         // Fixed vertices (1=fixed, 0=movable)
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;
      // Compute shader entry point for updating velocities before simulation
    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kBlockSize, 1, 1)  // 1D thread blocks for processing vertices
    void TrivialPBDClothVelocityUpdatePre(
        uint3 ThreadId : SV_DispatchThreadID
    )
    {
        uint index = ThreadId.x;
          // Early out if this thread is outside the cloth dimensions
        // We use a 1D dispatch for simplicity with velocity updates
        if (index >= PushConst.m_NumVertices)
            return;
            
        // Read current velocity and position
        float3 velocity = PushConst.m_VelocityBuffer.Load(index);
        float4 position = PushConst.m_PositionBuffer.Load(index);
        
        // Check if this vertex is fixed
        uint isFixed = PushConst.m_FixedVertexBuffer.Load(index);
        if (isFixed == 1)
        {
            // Fixed vertices don't move
            return;
        }
        
        // Get inverse mass from position buffer (w component)
        float invMass = position.w;
        
        // Skip vertices with zero inverse mass (immovable objects)
        if (invMass <= 0.0f)
            return;
            
        // Apply external forces (gravity)
        // F = m*a => a = F/m = F*invMass
        float3 gravityForce = float3(0.0f, -PushConst.m_Gravity, 0.0f); // Gravity in negative Y direction
        float3 acceleration = gravityForce * invMass;
        
        // Update velocity: v = v + a*dt
        velocity += acceleration * PushConst.m_DeltaTime;
        
        // Store updated velocity back to the buffer
        PushConst.m_VelocityBuffer.Store(velocity, index);
    }
    
} // namespace Siro
} // namespace IfritShader
