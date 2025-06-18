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
        uint m_NumVertices;             // Total number of vertices
        float m_DampingCoefficient;     // Basic velocity damping coefficient
        float m_RigidBodyDamping;       // Advanced rigid body damping (0=no damping, 1=full damping)
        float m_Epsilon;                // Small epsilon to avoid division by zero
        
        // Buffer handles
        TRWStructuredBufferHandle<float4> m_PositionBuffer;     // xyz=position, w=inv_mass
        TRWStructuredBufferHandle<float3> m_VelocityBuffer;     // Velocities
        TRWStructuredBufferHandle<SystemState> m_SystemStateBuffer; // Buffer for global system state
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;
    
    // Matrix operations for 3x3 matrices
    
    // Matrix multiplication: C = A * B
    void mat3x3_multiply(const float A[9], const float B[9], out float C[9])
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 3; k++) {
                    sum += A[i*3 + k] * B[k*3 + j];
                }
                C[i*3 + j] = sum;
            }
        }
    }
    
    // Matrix transpose: B = A^T
    void mat3x3_transpose(const float A[9], out float B[9])
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                B[j*3 + i] = A[i*3 + j];
            }
        }
    }
    
    // Matrix inversion: B = A^-1
    bool mat3x3_inverse(const float A[9], out float B[9])
    {
        float det = A[0] * (A[4] * A[8] - A[5] * A[7]) -
                    A[1] * (A[3] * A[8] - A[5] * A[6]) +
                    A[2] * (A[3] * A[7] - A[4] * A[6]);
                    
        // Check if determinant is too small
        if (abs(det) < PushConst.m_Epsilon)
            return false;
            
        float invDet = 1.0f / det;
        
        B[0] = (A[4] * A[8] - A[5] * A[7]) * invDet;
        B[1] = (A[2] * A[7] - A[1] * A[8]) * invDet;
        B[2] = (A[1] * A[5] - A[2] * A[4]) * invDet;
        B[3] = (A[5] * A[6] - A[3] * A[8]) * invDet;
        B[4] = (A[0] * A[8] - A[2] * A[6]) * invDet;
        B[5] = (A[2] * A[3] - A[0] * A[5]) * invDet;
        B[6] = (A[3] * A[7] - A[4] * A[6]) * invDet;
        B[7] = (A[1] * A[6] - A[0] * A[7]) * invDet;
        B[8] = (A[0] * A[4] - A[1] * A[3]) * invDet;
        
        return true;
    }
    
    // Compute a cross product matrix from a vector
    // This transforms a vector v to a skew-symmetric matrix [v]× such that [v]× * p = v × p
    void cross_product_matrix(float3 v, out float result[9])
    {
        result[0] = 0.0f;    result[1] = -v.z;   result[2] = v.y;
        result[3] = v.z;     result[4] = 0.0f;    result[5] = -v.x;
        result[6] = -v.y;    result[7] = v.x;     result[8] = 0.0f;
    }
    
    // Returns identity matrix
    void identity_matrix(out float result[9])
    {
        result[0] = 1.0f; result[1] = 0.0f; result[2] = 0.0f;
        result[3] = 0.0f; result[4] = 1.0f; result[5] = 0.0f;
        result[6] = 0.0f; result[7] = 0.0f; result[8] = 1.0f;
    }
    
    // Compute shader entry point for calculating system state for damping
    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)  // Single thread block that processes everything
    void TrivialPBDClothDampingSystemState(
        uint3 ThreadId : SV_DispatchThreadID
    )
    {
        // This entire shader runs as a single thread that accumulates properties
        // for the entire system
        if (ThreadId.x > 0 || ThreadId.y > 0 || ThreadId.z > 0)
            return;
            
        // Initialize system state values to zero
        float3 com = float3(0.0f, 0.0f, 0.0f);
        float3 comVelocity = float3(0.0f, 0.0f, 0.0f);
        float totalMass = 0.0f;
        
        // First pass: compute center of mass and total mass
        for (uint i = 0; i < PushConst.m_NumVertices; i++)
        {
            float4 posWithMass = PushConst.m_PositionBuffer.Load(i);
            float3 position = posWithMass.xyz;
            float invMass = posWithMass.w;
            
            // Skip fixed or zero-mass particles
            if (invMass <= 0.0f)
                continue;
                
            float mass = 1.0f / invMass;
            totalMass += mass;
            
            // Accumulate weighted position
            com += position * mass;
            
            // Accumulate weighted velocity
            float3 velocity = PushConst.m_VelocityBuffer.Load(i);
            comVelocity += velocity * mass;
        }
        
        // Normalize to get center of mass and velocity
        if (totalMass > PushConst.m_Epsilon)
        {
            com /= totalMass;
            comVelocity /= totalMass;
        }
        
        // Second pass: Compute angular momentum and inertia tensor
        float3 angularMomentum = float3(0.0f, 0.0f, 0.0f);
        float inertiaTensor[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        
        for (uint i = 0; i < PushConst.m_NumVertices; i++)
        {
            float4 posWithMass = PushConst.m_PositionBuffer.Load(i);
            float3 position = posWithMass.xyz;
            float invMass = posWithMass.w;
            
            // Skip fixed or zero-mass particles
            if (invMass <= 0.0f)
                continue;
                
            float mass = 1.0f / invMass;
            float3 velocity = PushConst.m_VelocityBuffer.Load(i);
            
            // Get position relative to center of mass
            float3 r = position - com;
            
            // Get velocity relative to center of mass
            float3 v = velocity - comVelocity;
            
            // Compute angular momentum: L += r × (m * v)
            angularMomentum += cross(r, v * mass);
            
            // Compute inertia tensor: I += m * (|r|^2 * I - r ⊗ r)
            float r_squared = dot(r, r);
            
            // Add m*|r|^2 to diagonal elements
            inertiaTensor[0] += mass * (r_squared - r.x * r.x);
            inertiaTensor[4] += mass * (r_squared - r.y * r.y);
            inertiaTensor[8] += mass * (r_squared - r.z * r.z);
            
            // Subtract m*r_i*r_j from off-diagonal elements
            inertiaTensor[1] -= mass * r.x * r.y;
            inertiaTensor[2] -= mass * r.x * r.z;
            inertiaTensor[5] -= mass * r.y * r.z;
        }
        
        // Copy symmetric elements
        inertiaTensor[3] = inertiaTensor[1]; // I_yx = I_xy
        inertiaTensor[6] = inertiaTensor[2]; // I_zx = I_xz
        inertiaTensor[7] = inertiaTensor[5]; // I_zy = I_yz
        
        // Compute inverse inertia tensor
        float invInertiaTensor[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        bool invertible = mat3x3_inverse(inertiaTensor, invInertiaTensor);
        
        // If matrix is not invertible, use identity matrix as a fallback
        if (!invertible)
        {
            identity_matrix(invInertiaTensor);
        }
        
        // Compute angular velocity: ω = I⁻¹ * L
        float3 angularVelocity = float3(
            invInertiaTensor[0] * angularMomentum.x + invInertiaTensor[1] * angularMomentum.y + invInertiaTensor[2] * angularMomentum.z,
            invInertiaTensor[3] * angularMomentum.x + invInertiaTensor[4] * angularMomentum.y + invInertiaTensor[5] * angularMomentum.z,
            invInertiaTensor[6] * angularMomentum.x + invInertiaTensor[7] * angularMomentum.y + invInertiaTensor[8] * angularMomentum.z
        );
        
        // Write computed values to the system state buffer
        SystemState state;
        state.centerOfMass = com;
        state.centerOfMassVelocity = comVelocity;
        state.angularMomentum = angularMomentum;
        state.totalMass = totalMass;
        state.angularVelocity = angularVelocity;
        
        // Copy inertia tensor and its inverse
        for (int i = 0; i < 9; i++) {
            state.inertiaTensor[i] = inertiaTensor[i];
            state.invInertiaTensor[i] = invInertiaTensor[i];
        }
        
        // Initialize padding (not necessary but good practice)
        state.padding[0] = 0;
        state.padding[1] = 0;
        state.padding[2] = 0;
        
        // Write the system state to the buffer
        PushConst.m_SystemStateBuffer.Store(state, 0);
    }
    
} // namespace Siro
} // namespace IfritShader