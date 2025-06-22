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

namespace IfritShader
{
    struct FMeshData
    {
        float4 m_BoundingSphere;
        TVertexDataHandle m_VertexBuffer;
        TNormalDataHandle m_NormalBuffer;
        TTangentDataHandle m_TangentBuffer;
        TUVDataHandle m_UVBuffer;
        uint m_MeshletBuffer;
        uint m_MeshletVertexBuffer;
        uint m_MeshletIndexBuffer;
        uint m_MeshletCullBuffer;
        uint m_BVHNodeBuffer;
        uint m_ClusterGroupBuffer;
        uint m_MeshletInClusterBuffer;
        uint m_CPCounterBuffer;
        uint m_MaterialDataId;
        uint m_IndexBuffer;
        uint m_Pad3;
    };

    struct FPerObjectData
    {
        TRWStructuredBufferHandle<FInstanceLocalTransform> m_Transform;
        TRWStructuredBufferHandle<FMeshData> m_MeshData;
        uint m_InstanceDataRef;
        TRWStructuredBufferHandle<FInstanceLocalTransform> m_TransformLast;
        uint m_MaterialId;
    };

}
