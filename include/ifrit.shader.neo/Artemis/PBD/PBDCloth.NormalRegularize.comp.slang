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

#include "ifrit.shader.neo/Artemis/PBD/PBDCloth.Shared.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T{
        TRWStructuredBufferHandle<float4> m_Normal;
        uint m_Vertices;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kArtemisTGSizeX,1,1)
    void ArtemisPBDClothNormalRegularizeCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_Vertices) return;

        float3 Normal = PushConst.m_Normal.Load(ThreadId.x).xyz;
        float Length = length(Normal);

        if (Length > 1e-6f) {
            Normal /= Length; // Normalize the normal vector
        } else {
            Normal = float3(0.0f, 0.0f, 1.0f); // Default normal if length is too small
        }

        PushConst.m_Normal.Store(float4(Normal, 0.0f), ThreadId.x);
    }
}
}