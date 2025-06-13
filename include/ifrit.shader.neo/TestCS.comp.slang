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

#include "ifrit.shader.neo/Bindless.hlsli" 
#include "ifrit.shader.neo/Math.MonteCarlo.hlsli"
#include "ifrit.shader.neo/Math.RayTrace.hlsli"
#include "ifrit.shader.neo/Math.SphericalHarmonics.hlsli"
#include "ifrit.shader.neo/Math.Transforms.hlsli"

#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
#include "ifrit.shader.neo/Lighting/CascadedShadowMapping.hlsli"

 
StructuredBuffer<float> buffer0;
StructuredBuffer<float> buffer1;
RWStructuredBuffer<float> result;

[shader("compute")]
[numthreads(1,1,1)]
void TestCS(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    result[index] = buffer0[index] + buffer1[index];
}

// Test