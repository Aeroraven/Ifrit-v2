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

namespace IfritShader {
namespace Siro {
    // Shared structures for TrivialPBDCloth shaders
    
    // SystemState struct used for the advanced damping algorithm
    // Stores global properties for the rigid body damping
    struct SystemState 
    {
        float3 centerOfMass;          // Center of mass position (x_cm)
        float3 centerOfMassVelocity;  // Center of mass velocity (v_cm)
        float3 angularMomentum;       // Angular momentum (L)
        float  inertiaTensor[9];      // 3x3 inertia tensor (I) stored as 9 floats in row-major order
        float  invInertiaTensor[9];   // 3x3 inverse inertia tensor (I^-1) stored as 9 floats in row-major order
        float3 angularVelocity;       // Angular velocity (ω)
        float  totalMass;             // Total mass of the system
        uint   padding[3];            // Padding to ensure alignment
    };
    
} // namespace Siro
} // namespace IfritShader
