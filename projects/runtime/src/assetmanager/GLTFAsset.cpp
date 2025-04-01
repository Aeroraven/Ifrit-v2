
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

#include "ifrit/runtime/assetmanager/GLTFAsset.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/algo/Hash.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/assetmanager/TextureAsset.h"
#include "ifrit/runtime/material/SyaroDefaultGBufEmitter.h"
#include <fstream>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "tinygltf2/tiny_gltf.h"

using Ifrit::SizeCast;

namespace Ifrit::Runtime
{

    struct GLTFInternalData
    {
        tinygltf::Model              model;
        Graphics::Rhi::RhiSamplerRef defaultSampler;
    };

    // Mesh class
    IFRIT_APIDECL GLTFPrefab::GLTFPrefab(IComponentManagerKeeper* keeper, AssetMetadata* metadata, GLTFAsset* asset,
        u32 meshId, u32 primitiveId, u32 nodeId, const Matrix4x4f& parentTransform)
        : m_asset(asset), m_meshId(meshId), m_primitiveId(primitiveId), m_nodeId(nodeId)
    {
        m_prefab = GameObject::CreatePrefab(keeper);

        auto  transform = m_prefab->GetComponent<Transform>();
        auto  data      = asset->GetInternalData(keeper);

        auto& gltfNode = data->model.nodes[nodeId];
        if (gltfNode.matrix.size())
        {
            iError("GLTFPrefab: matrix is not supported yet");
        }
        float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
        float scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;
        float rotX = 0.0f, rotY = 0.0f, rotZ = 0.0f, rotW = 1.0f;
        if (gltfNode.translation.size())
        {
            posX = static_cast<float>(gltfNode.translation[0]);
            posY = static_cast<float>(gltfNode.translation[1]);
            posZ = static_cast<float>(gltfNode.translation[2]);
        }
        if (gltfNode.scale.size())
        {
            scaleX = static_cast<float>(gltfNode.scale[0]);
            scaleY = static_cast<float>(gltfNode.scale[1]);
            scaleZ = static_cast<float>(gltfNode.scale[2]);
        }
        if (gltfNode.rotation.size())
        {
            rotX           = static_cast<float>(gltfNode.rotation[0]);
            rotY           = static_cast<float>(gltfNode.rotation[1]);
            rotZ           = static_cast<float>(gltfNode.rotation[2]);
            rotW           = static_cast<float>(gltfNode.rotation[3]);
            Vector3f euler = Math::QuaternionToEuler({ rotX, rotY, rotZ, rotW });
            rotX           = euler.x;
            rotY           = euler.y;
            rotZ           = euler.z;
        }

        auto localTransform =
            Math::GetTransformMatrix({ scaleX, scaleY, scaleZ }, { posX, posY, posZ }, { rotX, rotY, rotZ });
        auto     combinedTransform = Math::MatMul(parentTransform, localTransform);

        Vector3f newScale, newTranslation, newRotation;
        Math::RecoverTransformInfo(combinedTransform, newScale, newTranslation, newRotation);
        transform->SetPosition(newTranslation);
        transform->SetRotation(newRotation);
        transform->SetScale(newScale);
    }

    IFRIT_APIDECL Ref<MeshData> GLTFMesh::LoadMesh()
    {
        if (m_loaded)
        {
            return m_selfData;
        }
        // Start loading mesh
        m_selfData = std::make_shared<MeshData>();
        m_selfData->identifier =
            m_asset->GetMetadata().m_uuid + "_" + std::to_string(m_meshId) + "_" + std::to_string(m_primitiveId);

        auto& data      = m_asset->GetInternalDataForced()->model;
        auto& mesh      = data.meshes[m_meshId];
        auto& primitive = mesh.primitives[m_primitiveId];

        auto& accessorPos      = data.accessors[primitive.attributes["POSITION"]];
        auto& accessorNormal   = data.accessors[primitive.attributes["NORMAL"]];
        auto& accessorTexcoord = data.accessors[primitive.attributes["TEXCOORD_0"]];
        auto& accessorTangent  = data.accessors[primitive.attributes["TANGENT"]];
        auto& accessorIndices  = data.accessors[primitive.indices];

        bool  tangent3 = false;

        iAssertion(
            accessorPos.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT, "GLTFMesh: component type not supported");
        iAssertion(accessorPos.type == TINYGLTF_TYPE_VEC3, "GLTFMesh: type not supported");
        iAssertion(
            accessorNormal.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT, "GLTFMesh: component type not supported");
        iAssertion(accessorNormal.type == TINYGLTF_TYPE_VEC3, "GLTFMesh: type not supported");
        iAssertion(
            accessorTexcoord.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT, "GLTFMesh: component type not supported");
        iAssertion(accessorTexcoord.type == TINYGLTF_TYPE_VEC2, "GLTFMesh: type not supported");
        iAssertion(
            accessorTangent.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT, "GLTFMesh: component type not supported");
        if (accessorTangent.type == TINYGLTF_TYPE_VEC3)
        {
            tangent3 = true;
        }
        else
        {
            iAssertion(accessorTangent.type == TINYGLTF_TYPE_VEC4, "GLTFMesh: type not supported");
        }
        iAssertion(accessorIndices.type == TINYGLTF_TYPE_SCALAR, "GLTFMesh: type not supported");

        auto& posBufferView      = data.bufferViews[accessorPos.bufferView];
        auto& normalBufferView   = data.bufferViews[accessorNormal.bufferView];
        auto& texcoordBufferView = data.bufferViews[accessorTexcoord.bufferView];
        auto& indicesBufferView  = data.bufferViews[accessorIndices.bufferView];
        auto& tangentBufferView  = data.bufferViews[accessorTangent.bufferView];

        auto& posBuffer      = data.buffers[posBufferView.buffer];
        auto& normalBuffer   = data.buffers[normalBufferView.buffer];
        auto& texcoordBuffer = data.buffers[texcoordBufferView.buffer];
        auto& indicesBuffer  = data.buffers[indicesBufferView.buffer];
        auto& tangentBuffer  = data.buffers[tangentBufferView.buffer];

        auto  posData =
            reinterpret_cast<float*>(posBuffer.data.data() + posBufferView.byteOffset + accessorPos.byteOffset);
        auto normalData = reinterpret_cast<float*>(
            normalBuffer.data.data() + normalBufferView.byteOffset + accessorNormal.byteOffset);
        auto texcoordData = reinterpret_cast<float*>(
            texcoordBuffer.data.data() + texcoordBufferView.byteOffset + accessorTexcoord.byteOffset);
        auto tangentData = reinterpret_cast<float*>(
            tangentBuffer.data.data() + tangentBufferView.byteOffset + accessorTangent.byteOffset);
        auto indicesData = reinterpret_cast<u32*>(
            indicesBuffer.data.data() + indicesBufferView.byteOffset + accessorIndices.byteOffset);
        auto indicesDataUshort = reinterpret_cast<u16*>(
            indicesBuffer.data.data() + indicesBufferView.byteOffset + accessorIndices.byteOffset);

        auto posDataSize      = SizeCast<u32>(accessorPos.count * 3);
        auto normalDataSize   = SizeCast<u32>(accessorNormal.count * 3);
        auto texcoordDataSize = SizeCast<u32>(accessorTexcoord.count * 2);
        auto indicesDataSize  = SizeCast<u32>(accessorIndices.count);
        auto tangentStride    = tangent3 ? 3 : 4;
        auto tangentDataSize  = SizeCast<u32>(accessorTangent.count * tangentStride);

        m_selfData->m_vertices.resize(accessorPos.count);
        m_selfData->m_verticesAligned.resize(accessorPos.count);
        m_selfData->m_normals.resize(accessorNormal.count);
        m_selfData->m_normalsAligned.resize(accessorNormal.count);
        m_selfData->m_uvs.resize(accessorTexcoord.count);
        m_selfData->m_indices.resize(accessorIndices.count);
        m_selfData->m_tangents.resize(accessorTangent.count);

        Vector3f maxPos = { -1e10f, -1e10f, -1e10f };
        Vector3f minPos = { 1e10f, 1e10f, 1e10f };

        for (auto i = 0; i < accessorPos.count; i++)
        {
            m_selfData->m_vertices[i]        = { posData[i * 3], posData[i * 3 + 1], posData[i * 3 + 2] };
            m_selfData->m_verticesAligned[i] = { posData[i * 3], posData[i * 3 + 1], posData[i * 3 + 2], 1.0f };

            maxPos.x = std::max(maxPos.x, m_selfData->m_vertices[i].x);
            maxPos.y = std::max(maxPos.y, m_selfData->m_vertices[i].y);
            maxPos.z = std::max(maxPos.z, m_selfData->m_vertices[i].z);
            minPos.x = std::min(minPos.x, m_selfData->m_vertices[i].x);
            minPos.y = std::min(minPos.y, m_selfData->m_vertices[i].y);
            minPos.z = std::min(minPos.z, m_selfData->m_vertices[i].z);
        }
        for (auto i = 0; i < accessorNormal.count; i++)
        {
            m_selfData->m_normals[i]        = { normalData[i * 3], normalData[i * 3 + 1], -normalData[i * 3 + 2] };
            m_selfData->m_normalsAligned[i] = { normalData[i * 3], normalData[i * 3 + 1], normalData[i * 3 + 2], 0.0f };
        }
        for (auto i = 0; i < accessorTexcoord.count; i++)
        {
            m_selfData->m_uvs[i] = { texcoordData[i * 2], texcoordData[i * 2 + 1] };
        }
        for (auto i = 0; i < accessorTangent.count; i++)
        {
            m_selfData->m_tangents[i] = { tangentData[i * tangentStride], tangentData[i * tangentStride + 1],
                tangentData[i * tangentStride + 2], tangent3 ? 1.0f : tangentData[i * tangentStride + 3] };
        }

        if (accessorIndices.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
        {
            for (auto i = 0; i < accessorIndices.count; i++)
            {
                m_selfData->m_indices[i] = indicesDataUshort[i];
            }
        }
        else if (accessorIndices.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
        {
            for (auto i = 0; i < accessorIndices.count; i++)
            {
                m_selfData->m_indices[i] = indicesData[i];
            }
        }
        else
        {
            iError("GLTFMesh: index component type not supported");
        }
        m_selfData->m_BoundingBoxMax = maxPos;
        m_selfData->m_BoundingBoxMin = minPos;

        this->CreateMeshLodHierarchy(m_selfData, m_cachePath);
        m_loaded      = true;
        m_selfDataRaw = m_selfData.get();
        return m_selfData;
    }

    IFRIT_APIDECL MeshData* GLTFMesh::LoadMeshUnsafe()
    {
        if (m_loaded)
        {
            return m_selfDataRaw;
        }
        else
        {
            LoadMesh();
            m_selfDataRaw = m_selfData.get();
            return m_selfDataRaw;
        }
    }

    // Asset class

    IFRIT_APIDECL void GLTFAsset::RequestLoad(IComponentManagerKeeper* keeper)
    {
        if (!m_loaded)
        {
            m_loaded = true;
            LoadGLTF(m_manager, keeper);
        }
    }

    IFRIT_APIDECL void GLTFAsset::LoadGLTF(AssetManager* m_manager, IComponentManagerKeeper* keeper)
    {
        m_internalData = new GLTFInternalData();
        auto rhi       = m_manager->GetApplication()->GetRhi();
        if (m_internalData->defaultSampler == nullptr)
        {
            m_internalData->defaultSampler = rhi->CreateTrivialBilinearSampler(true);
        }
        tinygltf::TinyGLTF loader;
        String             err;
        String             warn;
        bool               ret = loader.LoadASCIIFromFile(&m_internalData->model, &err, &warn, m_path.string());
        if (!warn.empty())
        {
            iWarn("GLTFAsset: {}", warn);
        }
        if (!err.empty())
        {
            iError("GLTFAsset: {}", err);
        }
        // Create meshes
        std::unordered_map<std::pair<u32, u32>, u32, PairwiseHash<u32, u32>> meshHash;
        auto cachePath = m_manager->GetApplication()->GetCacheDir();
        for (auto i = 0; auto& mesh : m_internalData->model.meshes)
        {
            for (auto j = 0; auto& primitive : mesh.primitives)
            {
                auto mesh = std::make_shared<GLTFMesh>(&m_metadata, this, i, j, ~0u, cachePath);
                m_meshes.push_back(std::move(mesh));
                meshHash[{ i, j }] = m_meshes.size() - 1;
                j++;
            }
            i++;
        }

        // Create prefab for each node
        // get gltf path
        auto                      gltfDir = m_path.parent_path();

        auto                      rawGLTFData = m_internalData->model;

        Fn<void(int, Matrix4x4f)> traverseNode = [&](int nodeId, Matrix4x4f parentTransform) {
            auto& node         = rawGLTFData.nodes[nodeId];
            auto  translationX = 0.0f, translationY = 0.0f, translationZ = 0.0f;
            auto  scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;
            auto  rotX = 0.0f, rotY = 0.0f, rotZ = 0.0f, rotW = 1.0f;
            if (node.translation.size())
            {
                translationX = static_cast<float>(node.translation[0]);
                translationY = static_cast<float>(node.translation[1]);
                translationZ = static_cast<float>(node.translation[2]);
            }
            if (node.scale.size())
            {
                scaleX = static_cast<float>(node.scale[0]);
                scaleY = static_cast<float>(node.scale[1]);
                scaleZ = static_cast<float>(node.scale[2]);
            }
            if (node.rotation.size())
            {
                rotX = static_cast<float>(node.rotation[0]);
                rotY = static_cast<float>(node.rotation[1]);
                rotZ = static_cast<float>(node.rotation[2]);
                rotW = static_cast<float>(node.rotation[3]);
                if (node.rotation.size() == 4)
                {
                    Vector3f euler = Math::QuaternionToEuler({ rotX, rotY, rotZ, rotW });
                    rotX           = euler.x;
                    rotY           = euler.y;
                    rotZ           = euler.z;
                }
            }
            auto childTransform = Math::GetTransformMatrix(
                { scaleX, scaleY, scaleZ }, { translationX, translationY, translationZ }, { rotX, rotY, rotZ });

            // Check children
            if (node.children.size())
            {
                for (auto& childId : node.children)
                {
                    traverseNode(childId, Math::MatMul(parentTransform, childTransform));
                }
            }
            // if mesh is present
            if (node.mesh >= 0)
            {
                for (auto j = 0; auto& primitive : rawGLTFData.meshes[node.mesh].primitives)
                {
                    auto prefab =
                        std::make_shared<GLTFPrefab>(keeper, &m_metadata, this, node.mesh, j, nodeId, parentTransform);
                    auto meshFilter = prefab->m_prefab->AddComponent<MeshFilter>();
                    auto mesh       = m_meshes[meshHash[{ SizeCast<u32>(node.mesh), j }]];
                    meshFilter->SetMesh(mesh);

                    auto& gltfMaterial      = rawGLTFData.materials[primitive.material];
                    auto& normalData        = gltfMaterial.normalTexture;
                    auto& pbrData           = gltfMaterial.pbrMetallicRoughness;
                    auto  baseColorIndexTex = pbrData.baseColorTexture.index;
                    if (baseColorIndexTex == -1)
                    {
                        // find diffuse texture
                        auto extensionExists = gltfMaterial.extensions.find("KHR_materials_pbrSpecularGlossiness");
                        if (extensionExists != gltfMaterial.extensions.end())
                        {
                            auto& extension     = gltfMaterial.extensions["KHR_materials_pbrSpecularGlossiness"];
                            auto  diffuseExists = extension.Has("diffuseTexture");
                            if (diffuseExists)
                            {
                                auto& diffuseTexture = extension.Get("diffuseTexture");
                                baseColorIndexTex    = diffuseTexture.Get("index").Get<i32>();
                            }
                        }
                    }
                    auto normalTexIndexTex = normalData.index;
                    auto baseColorIndex    = rawGLTFData.textures[baseColorIndexTex].source;
                    auto normalTexIndex    = rawGLTFData.textures[normalTexIndexTex].source;
                    auto baseColorURI      = rawGLTFData.images[baseColorIndex].uri;
                    auto normalTexURI      = rawGLTFData.images[normalTexIndex].uri;

                    auto texPathBase = gltfDir / baseColorURI;
                    auto texBase     = m_manager->requestAsset<Asset>(texPathBase);
                    auto material    = std::make_shared<SyaroDefaultGBufEmitter>(m_manager->GetApplication());

                    auto texCastedBase = std::dynamic_pointer_cast<TextureAsset>(texBase);
                    if (!texCastedBase)
                    {
                        iWarn("GLTFAsset: invalid texture asset");
                    }
                    else
                    {
                        auto albedoId = rhi->RegisterCombinedImageSampler(
                            texCastedBase->GetTexture().get(), m_internalData->defaultSampler.get());
                        material->SetAlbedoId(albedoId->GetActiveId());
                    }

                    auto texPathNormal   = gltfDir / normalTexURI;
                    auto texNormal       = m_manager->requestAsset<Asset>(texPathNormal);
                    auto texCastedNormal = std::dynamic_pointer_cast<TextureAsset>(texNormal);
                    if (!texCastedNormal)
                    {
                        iWarn("GLTFAsset: invalid texture asset");
                    }
                    else
                    {
                        auto normalId = rhi->RegisterCombinedImageSampler(
                            texCastedNormal->GetTexture().get(), m_internalData->defaultSampler.get());
                        material->SetNormalMapId(normalId->GetActiveId());
                    }

                    auto meshRenderer = prefab->m_prefab->AddComponent<MeshRenderer>();
                    material->BuildMaterial();
                    meshRenderer->SetMaterial(material);
                    m_prefabs.push_back(std::move(prefab));
                    j++;
                }
            }
        };

        // Get nodes from scenes
        for (auto& scene : rawGLTFData.scenes)
        {
            for (auto& nodeId : scene.nodes)
            {
                traverseNode(nodeId, Math::Identity4());
            }
        }
    }

    IFRIT_APIDECL GLTFInternalData* GLTFAsset::GetInternalData(IComponentManagerKeeper* keeper)
    {
        RequestLoad(keeper);
        return m_internalData;
    }

    IFRIT_APIDECL GLTFInternalData* GLTFAsset::GetInternalDataForced() { return m_internalData; }

    IFRIT_APIDECL                   GLTFAsset::~GLTFAsset()
    {
        if (m_internalData)
        {
            delete m_internalData;
        }
    }

    // Importer
    IFRIT_APIDECL void GLTFAssetImporter::ProcessMetadata(AssetMetadata& metadata)
    {
        metadata.m_importer = IMPORTER_NAME;
    }

    IFRIT_APIDECL std::vector<String> GLTFAssetImporter::GetSupportedExtensionNames() { return { ".gltf", ".glb" }; }

    IFRIT_APIDECL void GLTFAssetImporter::ImportAsset(const std::filesystem::path& path, AssetMetadata& metadata)
    {
        auto asset = std::make_shared<GLTFAsset>(metadata, path, m_assetManager);
        m_assetManager->RegisterAsset(asset);

        // iInfo("Imported asset: [GLTFObject] {}", metadata.m_uuid);
    }

} // namespace Ifrit::Runtime