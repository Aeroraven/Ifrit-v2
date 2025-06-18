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
        uint m_Width;                // Cloth width (vertices)
        uint m_Height;               // Cloth height (vertices)
        float m_InvMass;             // Inverse mass (1/mass)
        float m_Spacing;             // Spacing between vertices
        float m_DampingCoefficient;  // Velocity damping
        float m_Gravity;             // Gravity force
        uint m_NumStretchConstraints; // Number of stretch constraints
        uint m_NumBendConstraints;    // Number of bend constraints
        
        // Buffer handles
        TRWStructuredBufferHandle<float4> m_PositionBuffer;        // xyz=position, w=inv_mass
        TRWStructuredBufferHandle<float4> m_PrevPositionBuffer;    // Previous positions
        TRWStructuredBufferHandle<float3> m_VelocityBuffer;        // Velocities
        TRWStructuredBufferHandle<float3> m_OffsetBuffer;          // Position adjustment offsets
        TRWStructuredBufferHandle<uint> m_FixedVertexBuffer;       // Fixed vertices (1=fixed, 0=movable)
        TRWStructuredBufferHandle<uint3> m_IndexBuffer;            // Triangle indices
        
        // Optional: Constraint buffers can be initialized in a separate pass if needed
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;
    
    // Compute shader entry point for initializing the cloth simulation
    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kBlockSize, kBlockSize, 1)
    void TrivialPBDClothInit(
        uint3 ThreadId : SV_DispatchThreadID,
        uint3 GroupThreadId : SV_GroupThreadID,
        uint3 GroupId : SV_GroupID
    )
    {
        // Calculate vertex index from thread ID
        uint x = ThreadId.x;
        uint y = ThreadId.y;
        
        // Early out if this thread is outside the cloth dimensions
        if (x >= PushConst.m_Width || y >= PushConst.m_Height)
            return;
            
        uint vertexIndex = y * PushConst.m_Width + x;
        
        // Calculate initial position for the vertex
        // Create a grid centered at (0,0,0)
        float halfWidth = (PushConst.m_Width - 1) * PushConst.m_Spacing * 0.5f;
        float halfHeight = (PushConst.m_Height - 1) * PushConst.m_Spacing * 0.5f;
        
        float3 position = float3(
            x * PushConst.m_Spacing - halfWidth, 
            0.0f,  // Start flat on xz plane (y is up)
            y * PushConst.m_Spacing - halfHeight
        );
        
        // Determine if this is a fixed vertex (typically corners at the top)
        bool isFixed = (y == 0) && (x == 0 || x == PushConst.m_Width - 1);
        float invMass = isFixed ? 0.0f : PushConst.m_InvMass;  // Zero inv_mass means immovable
        
        // Initialize position buffer with xyz=position, w=inv_mass
        PushConst.m_PositionBuffer.Store(float4(position, invMass), vertexIndex);
        
        // Initialize previous position to the same as current position
        PushConst.m_PrevPositionBuffer.Store(float4(position, invMass), vertexIndex);
        
        // Initialize velocity to zero
        PushConst.m_VelocityBuffer.Store(float3(0.0f, 0.0f, 0.0f), vertexIndex);
        
        // Initialize offset to zero
        PushConst.m_OffsetBuffer.Store(float3(0.0f, 0.0f, 0.0f), vertexIndex);
        
        // Mark fixed vertices
        PushConst.m_FixedVertexBuffer.Store(isFixed ? 1 : 0, vertexIndex);
        
        // Generate triangle indices (only first thread in each 2x2 block)
        // Each quad generates 2 triangles
        if (x < PushConst.m_Width - 1 && y < PushConst.m_Height - 1)
        {
            uint topLeft = y * PushConst.m_Width + x;
            uint topRight = topLeft + 1;
            uint bottomLeft = (y + 1) * PushConst.m_Width + x;
            uint bottomRight = bottomLeft + 1;
            
            uint triangleIndex = (y * (PushConst.m_Width - 1) + x) * 2;
            
            // First triangle (top-left, bottom-left, bottom-right)
            PushConst.m_IndexBuffer.Store(uint3(topLeft, bottomLeft, bottomRight), triangleIndex);
            
            // Second triangle (top-left, bottom-right, top-right)
            PushConst.m_IndexBuffer.Store(uint3(topLeft, bottomRight, topRight), triangleIndex + 1);
        }
    }
    
} // namespace Siro
} // namespace IfritShader
