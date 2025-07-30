
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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

#include "ifrit/runtime/asset/mesh/WaveFrontAsset.h"
#include "ifrit/runtime/common/Pch.h"
#include <fstream>

using Ifrit::SizeCast;

namespace Ifrit::Runtime
{

    void LoadWaveFrontObject(
        const char* path, Vec<Vector3f>& vertices, Vec<Vector3f>& normals, Vec<Vector2f>& uvs, Vec<u32>& indices)
    {

        // This section is auto-generated from Copilot
        std::ifstream file(path);
        String        line;
        Vec<u32>      interIdx;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            String             type;
            iss >> type;

            if (type == "v")
            {
                Vector3f vertex;
                iss >> vertex.x >> vertex.y >> vertex.z;
                vertices.push_back(vertex);
            }
            else if (type == "vn")
            {
                Vector3f normal;
                iss >> normal.x >> normal.y >> normal.z;
                normals.push_back(normal);
            }
            else if (type == "vt")
            {
                Vector2f uv;
                iss >> uv.x >> uv.y;
                uvs.push_back(uv);
            }
            else if (type == "f")
            {
                String vertex;
                for (int i = 0; i < 3; i++)
                {
                    iss >> vertex;
                    std::istringstream vss(vertex);
                    String             index;
                    for (int j = 0; j < 3; j++)
                    {
                        std::getline(vss, index, '/');
                        if (index.size() != 0)
                        {
                            interIdx.push_back(std::stoi(index) - 1);
                        }
                        else
                        {
                            interIdx.push_back(0);
                        }
                    }
                }
            }
        }
        indices.resize(interIdx.size());
        for (int i = 0; i < interIdx.size(); i++)
        {
            indices[i] = interIdx[i];
        }
    }
    Vec<Vector3f> RemapNormals(Vec<Vector3f> normals, Vec<u32> indices, int numVertices)
    {
        using namespace Ifrit::Math;
        Vec<Vector3f> retNormals;
        Vec<int>      counters;
        retNormals.clear();
        counters.clear();
        retNormals.resize(numVertices);
        counters.resize(numVertices);
        for (int i = 0; i < numVertices; i++)
        {
            retNormals[i] = { 0, 0, 0 };
            counters[i]   = 0;
        }
        for (int i = 0; i < indices.size(); i += 3)
        {
            auto faceNode   = indices[i];
            auto normalNode = indices[i + 2];
            retNormals[faceNode].x += normals[normalNode].x;
            retNormals[faceNode].y += normals[normalNode].y;
            retNormals[faceNode].z += normals[normalNode].z;
            counters[faceNode]++;
        }
        for (int i = 0; i < numVertices; i++)
        {
            retNormals[i] = Normalize(retNormals[i]);
        }
        return retNormals;
    }

    Vec<Vector2f> RemapUVs(Vec<Vector2f> uvs, Vec<u32> indices, int numVertices)
    {
        Vec<Vector2f> retNormals;
        Vec<int>      counters;
        retNormals.clear();
        counters.clear();
        retNormals.resize(numVertices);
        counters.resize(numVertices);
        for (int i = 0; i < numVertices; i++)
        {
            retNormals[i] = { 0, 0 };
            counters[i]   = 0;
        }
        for (int i = 0; i < indices.size(); i += 3)
        {
            auto faceNode          = indices[i];
            auto normalNode        = indices[i + 1];
            retNormals[faceNode].x = uvs[normalNode].x;
            retNormals[faceNode].y = uvs[normalNode].y;
            counters[faceNode]++;
        }
        return retNormals;
    }
    // Mesh class

    IFRIT_APIDECL Ref<MeshData> ImportedWaveFrontMesh::LoadMesh()
    {
        if (m_loaded)
        {
            return m_selfData;
        }
        else
        {
            m_loaded   = true;
            m_selfData = MakeRef<MeshData>();
            Vec<Vector3f> vertices;
            Vec<Vector3f> normals;
            Vec<Vector3f> remappedNormals;
            Vec<Vector2f> uvs;
            Vec<Vector2f> remappedUVs;
            Vec<u32>      remappedIndices;
            Vec<u32>      indices;
            auto          rawPath = mResourcePath;
            LoadWaveFrontObject(rawPath.c_str(), vertices, normals, uvs, indices);
            remappedNormals = RemapNormals(normals, indices, SizeCast<int>(vertices.size()));
            if (uvs.size() != 0)
            {
                remappedUVs = RemapUVs(uvs, indices, SizeCast<int>(vertices.size()));
            }
            else
            {
                remappedUVs.resize(vertices.size());
            }

            m_selfData->m_vertices = vertices;
            m_selfData->m_normals  = remappedNormals;
            m_selfData->m_uvs      = remappedUVs;

            // remap indices
            remappedIndices.resize(indices.size() / 3);
            for (int i = 0; i < indices.size(); i += 3)
            {
                remappedIndices[i / 3] = indices[i];
            }
            m_selfData->m_indices = remappedIndices;

            // align vertices
            m_selfData->m_verticesAligned.resize(vertices.size());
            m_selfData->m_normalsAligned.resize(vertices.size());
            m_selfData->m_tangents.resize(vertices.size());
            for (int i = 0; i < vertices.size(); i++)
            {
                m_selfData->m_verticesAligned[i] = Vector4f(vertices[i].x, vertices[i].y, vertices[i].z, 1.0);
                m_selfData->m_normalsAligned[i] =
                    Vector4f(remappedNormals[i].x, remappedNormals[i].y, remappedNormals[i].z, 1.0);
                m_selfData->m_tangents[i] = Vector4f(0.0f, 0.0f, 0.0f, 1.0f); // Default tangent, can be set later
            }
            this->CreateMeshLodHierarchy(m_selfData, "");
        }
        return m_selfData;
    }

    IFRIT_APIDECL MeshData* ImportedWaveFrontMesh::LoadMeshUnsafe()
    {
        if (m_selfDataRaw == nullptr)
        {
            if (m_selfData == nullptr)
            {
                m_selfDataRaw = LoadMesh().get();
            }
            else
            {
                m_selfDataRaw = m_selfData.get();
            }
        }
        return m_selfDataRaw;
    }
    IFRIT_APIDECL Mesh* WaveFrontAsset::GetMesh()
    {

        if (mImportedMesh == nullptr)
        {
            mImportedMesh                = MakeOwner<ImportedWaveFrontMesh>();
            mImportedMesh->mResourcePath = mMetadata.mExternalPath;
            mImportedMesh->LoadMeshUnsafe();
        }
        return mImportedMesh.get();
    }

    IFRIT_APIDECL u32 ImportedWaveFrontMesh::GetNumIndices() { return SizeCast<u32>(m_selfData->m_indices.size()); }
    IFRIT_APIDECL u32 ImportedWaveFrontMesh::GetNumVertices() { return SizeCast<u32>(m_selfData->m_vertices.size()); }
    IFRIT_APIDECL Vec<u32> ImportedWaveFrontMesh::GetIndexBufferHost() { return m_selfData->m_indices; }
    IFRIT_APIDECL Vec<Vector3f> ImportedWaveFrontMesh::GetVertexBufferHost() { return m_selfData->m_vertices; }

} // namespace Ifrit::Runtime