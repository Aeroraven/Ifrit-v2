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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "ifrit/runtime/physics/artemis/geometry/TessellatedRectMesh.h"

namespace Ifrit::Runtime::Artemis
{

    // virtual Ref<MeshData> LoadMesh() override;
    // virtual MeshData*     LoadMeshUnsafe() override;
    // virtual u32           GetNumIndices() override;
    // virtual u32           GetNumVertices() override;
    // virtual Vec<u32>      GetIndexBufferHost() override;
    // virtual Vec<Vector3f> GetVertexBufferHost() override;

    IFRIT_APIDECL TessellatedRectMesh::TessellatedRectMesh(
        f32 width, f32 height, u32 tessRows, u32 tessCols, Vector3f translate)
        : Mesh(), m_Width(width), m_Height(height), m_TessRows(tessRows), m_TessCols(tessCols), m_Translate(translate)
    {
        Initialize();
    }

    IFRIT_APIDECL void TessellatedRectMesh::Initialize()
    {
        m_SelfData             = std::make_shared<MeshData>();
        m_SelfData->m_MeshType = MeshType::Surface;
    }

    IFRIT_APIDECL Ref<MeshData> TessellatedRectMesh::LoadMesh()
    {
        if (m_Loaded)
        {
            return m_SelfData;
        }
        BuildTopology();
        m_Loaded = true;
        return m_SelfData;
    }

    IFRIT_APIDECL MeshData* TessellatedRectMesh::LoadMeshUnsafe()
    {
        if (m_Loaded)
        {
            return m_SelfData.get();
        }
        BuildTopology();
        m_Loaded = true;
        return m_SelfData.get();
    }

    IFRIT_APIDECL u32 TessellatedRectMesh::GetNumIndices() {return SizeCast<u32>(m_SelfData->m_indices.size()); }
    IFRIT_APIDECL u32 TessellatedRectMesh::GetNumVertices() {return SizeCast<u32>(m_SelfData->m_vertices.size()); }
    IFRIT_APIDECL Vec<u32> TessellatedRectMesh::GetIndexBufferHost()
    {
        if (m_Loaded)
        {
            return m_SelfData->m_indices;
        }
        else
        {
            LoadMesh();
            return m_SelfData->m_indices;
        }
    }

    IFRIT_APIDECL Vec<Vector3f> TessellatedRectMesh::GetVertexBufferHost()
    {
        if (m_Loaded)
        {
            return m_SelfData->m_vertices;
        }
        else
        {
            LoadMesh();
            return m_SelfData->m_vertices;
        }
    }

    IFRIT_APIDECL void TessellatedRectMesh::BuildTopology()
    {
        m_SelfData->m_vertices.clear();
        m_SelfData->m_indices.clear();

        f32 stepX = m_Width / static_cast<f32>(m_TessCols);
        f32 stepY = m_Height / static_cast<f32>(m_TessRows);

        for (u32 row = 0; row <= m_TessRows; ++row)
        {
            for (u32 col = 0; col <= m_TessCols; ++col)
            {
                Vector3f vertex = { col * stepX + m_Translate.x, m_Translate.y, row * stepY + m_Translate.z };
                m_SelfData->m_vertices.push_back(vertex);
                m_SelfData->m_verticesAligned.push_back(
                    Vector4f(vertex.x, vertex.y, vertex.z, 1.0f)); // Assuming w = 1.0 for homogeneous coordinates

                // Normal use (0,0,1)
                Vector3f normal = { 0.0f, 1.0f, 0.0f };
                m_SelfData->m_normals.push_back(normal);
                m_SelfData->m_normalsAligned.push_back(
                    Vector4f(normal.x, normal.y, normal.z, 0.0f)); // Assuming w = 0.0 for normals

                // UV coordinates
                Vector2f uv = { static_cast<f32>(col) / m_TessCols, static_cast<f32>(row) / m_TessRows };
                m_SelfData->m_uvs.push_back(uv);
                m_SelfData->m_tangents.push_back(Vector4f(1.0f, 0.0f, 0.0f, 0.0f)); // Assuming tangent (1,0,0,0)
            }
        }

        u32 triangleType = 0;

        for (u32 row = 0; row < m_TessRows; ++row)
        {
            for (u32 col = 0; col < m_TessCols; ++col)
            {
                u32 index1 = row * (m_TessCols + 1) + col;
                u32 index2 = index1 + 1;
                u32 index3 = index1 + (m_TessCols + 1);
                u32 index4 = index3 + 1;
                if (triangleType == 0)
                {
                    // Two triangles per quad
                    m_SelfData->m_indices.push_back(index1);
                    m_SelfData->m_indices.push_back(index2);
                    m_SelfData->m_indices.push_back(index3);

                    m_SelfData->m_indices.push_back(index2);
                    m_SelfData->m_indices.push_back(index4);
                    m_SelfData->m_indices.push_back(index3);
                }
                else
                {
                    // 124
                    m_SelfData->m_indices.push_back(index1);
                    m_SelfData->m_indices.push_back(index2);
                    m_SelfData->m_indices.push_back(index4);

                    // 134
                    m_SelfData->m_indices.push_back(index1);
                    m_SelfData->m_indices.push_back(index4);
                    m_SelfData->m_indices.push_back(index3);
                }
            }
        }
    }

} // namespace Ifrit::Runtime::Artemis