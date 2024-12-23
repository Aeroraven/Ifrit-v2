
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

#include "ifrit/core/assetmanager/GLTFAsset.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/util/Hash.h"
#include "ifrit/common/util/TypingUtil.h"
#include <fstream>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "tinygltf2/tiny_gltf.h"

using Ifrit::Common::Utility::size_cast;

namespace Ifrit::Core {

struct GLTFInternalData {
  tinygltf::Model model;
};

// Mesh class
IFRIT_APIDECL GLTFPrefab::GLTFPrefab(AssetMetadata *metadata, GLTFAsset *asset,
                                     uint32_t meshId, uint32_t primitiveId,
                                     uint32_t nodeId)
    : m_asset(asset), m_meshId(meshId), m_primitiveId(primitiveId),
      m_nodeId(nodeId) {
  m_prefab = SceneObject::createPrefab();

  auto transform = m_prefab->getComponent<Transform>();
  auto data = asset->getInternalData();

  auto &gltfNode = data->model.nodes[nodeId];
  if (gltfNode.matrix.size()) {
    iError("GLTFPrefab: matrix is not supported yet");
  }
  if (gltfNode.translation.size()) {
    float posX = static_cast<float>(gltfNode.translation[0]);
    float posY = static_cast<float>(gltfNode.translation[1]);
    float posZ = static_cast<float>(gltfNode.translation[2]);
    transform->setPosition({posX, posY, posZ});
  }
  if (gltfNode.scale.size()) {
    float scaleX = static_cast<float>(gltfNode.scale[0]);
    float scaleY = static_cast<float>(gltfNode.scale[1]);
    float scaleZ = static_cast<float>(gltfNode.scale[2]);
    transform->setScale({scaleX, scaleY, scaleZ});
  }
  if (gltfNode.rotation.size()) {
    float rotX = static_cast<float>(gltfNode.rotation[0]);
    float rotY = static_cast<float>(gltfNode.rotation[1]);
    float rotZ = static_cast<float>(gltfNode.rotation[2]);
    float rotW = static_cast<float>(gltfNode.rotation[3]);
    ifloat3 euler = Math::quaternionToEuler({rotX, rotY, rotZ, rotW});
    // transform->setRotation(euler);
  }
}

IFRIT_APIDECL std::shared_ptr<MeshData> GLTFMesh::loadMesh() {
  if (m_loaded) {
    return m_selfData;
  }
  // Start loading mesh
  m_selfData = std::make_shared<MeshData>();
  auto &data = m_asset->getInternalData()->model;
  auto &mesh = data.meshes[m_meshId];
  auto &primitive = mesh.primitives[m_primitiveId];

  auto &accessorPos = data.accessors[primitive.attributes["POSITION"]];
  auto &accessorNormal = data.accessors[primitive.attributes["NORMAL"]];
  auto &accessorTexcoord = data.accessors[primitive.attributes["TEXCOORD_0"]];
  auto &accessorIndices = data.accessors[primitive.indices];

  iAssertion(accessorPos.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT,
             "GLTFMesh: component type not supported");
  iAssertion(accessorPos.type == TINYGLTF_TYPE_VEC3,
             "GLTFMesh: type not supported");
  iAssertion(accessorNormal.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT,
             "GLTFMesh: component type not supported");
  iAssertion(accessorNormal.type == TINYGLTF_TYPE_VEC3,
             "GLTFMesh: type not supported");
  iAssertion(accessorTexcoord.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT,
             "GLTFMesh: component type not supported");
  iAssertion(accessorTexcoord.type == TINYGLTF_TYPE_VEC2,
             "GLTFMesh: type not supported");
  iAssertion(accessorIndices.type == TINYGLTF_TYPE_SCALAR,
             "GLTFMesh: type not supported");

  auto &posBufferView = data.bufferViews[accessorPos.bufferView];
  auto &normalBufferView = data.bufferViews[accessorNormal.bufferView];
  auto &texcoordBufferView = data.bufferViews[accessorTexcoord.bufferView];
  auto &indicesBufferView = data.bufferViews[accessorIndices.bufferView];

  auto &posBuffer = data.buffers[posBufferView.buffer];
  auto &normalBuffer = data.buffers[normalBufferView.buffer];
  auto &texcoordBuffer = data.buffers[texcoordBufferView.buffer];
  auto &indicesBuffer = data.buffers[indicesBufferView.buffer];

  auto posData = reinterpret_cast<float *>(posBuffer.data.data() +
                                           posBufferView.byteOffset +
                                           accessorPos.byteOffset);
  auto normalData = reinterpret_cast<float *>(normalBuffer.data.data() +
                                              normalBufferView.byteOffset +
                                              accessorNormal.byteOffset);
  auto texcoordData = reinterpret_cast<float *>(texcoordBuffer.data.data() +
                                                texcoordBufferView.byteOffset +
                                                accessorTexcoord.byteOffset);
  auto indicesData = reinterpret_cast<uint32_t *>(indicesBuffer.data.data() +
                                                  indicesBufferView.byteOffset +
                                                  accessorIndices.byteOffset);
  auto indicesDataUshort = reinterpret_cast<uint16_t *>(
      indicesBuffer.data.data() + indicesBufferView.byteOffset +
      accessorIndices.byteOffset);

  auto posDataSize = size_cast<uint32_t>(accessorPos.count * 3);
  auto normalDataSize = size_cast<uint32_t>(accessorNormal.count * 3);
  auto texcoordDataSize = size_cast<uint32_t>(accessorTexcoord.count * 2);
  auto indicesDataSize = size_cast<uint32_t>(accessorIndices.count);

  m_selfData->m_vertices.resize(accessorPos.count);
  m_selfData->m_verticesAligned.resize(accessorPos.count);
  m_selfData->m_normals.resize(accessorNormal.count);
  m_selfData->m_normalsAligned.resize(accessorNormal.count);
  m_selfData->m_uvs.resize(accessorTexcoord.count);
  m_selfData->m_indices.resize(accessorIndices.count);

  for (auto i = 0; i < accessorPos.count; i++) {
    m_selfData->m_vertices[i] = {posData[i * 3], posData[i * 3 + 1],
                                 posData[i * 3 + 2]};
    m_selfData->m_verticesAligned[i] = {posData[i * 3], posData[i * 3 + 1],
                                        posData[i * 3 + 2], 1.0f};
  }
  for (auto i = 0; i < accessorNormal.count; i++) {
    m_selfData->m_normals[i] = {normalData[i * 3], normalData[i * 3 + 1],
                                normalData[i * 3 + 2]};
    m_selfData->m_normalsAligned[i] = {normalData[i * 3], normalData[i * 3 + 1],
                                       normalData[i * 3 + 2], 0.0f};
  }
  for (auto i = 0; i < accessorTexcoord.count; i++) {
    m_selfData->m_uvs[i] = {texcoordData[i * 2], texcoordData[i * 2 + 1]};
  }
  if (accessorIndices.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
    for (auto i = 0; i < accessorIndices.count; i++) {
      m_selfData->m_indices[i] = indicesDataUshort[i];
    }
  } else if (accessorIndices.componentType ==
             TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
    for (auto i = 0; i < accessorIndices.count; i++) {
      m_selfData->m_indices[i] = indicesData[i];
    }
  } else {
    iError("GLTFMesh: index component type not supported");
  }
  this->createMeshLodHierarchy(m_selfData);
  m_loaded = true;
  m_selfDataRaw = m_selfData.get();
  return m_selfData;
}

IFRIT_APIDECL MeshData *GLTFMesh::loadMeshUnsafe() {
  if (m_loaded) {
    return m_selfDataRaw;
  } else {
    loadMesh();
    m_selfDataRaw = m_selfData.get();
    return m_selfDataRaw;
  }
}

// Asset class

IFRIT_APIDECL void GLTFAsset::loadGLTF() {
  m_internalData = new GLTFInternalData();
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;
  bool ret = loader.LoadASCIIFromFile(&m_internalData->model, &err, &warn,
                                      m_path.string());
  if (!warn.empty()) {
    iWarn("GLTFAsset: {}", warn);
  }
  if (!err.empty()) {
    iError("GLTFAsset: {}", err);
  }
  // Create meshes
  std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t,
                     Ifrit::Common::Utility::PairwiseHash<uint32_t, uint32_t>>
      meshHash;
  for (auto i = 0; auto &mesh : m_internalData->model.meshes) {
    for (auto j = 0; auto &primitive : mesh.primitives) {
      auto mesh = std::make_shared<GLTFMesh>(&m_metadata, this, i, j, ~0u);
      m_meshes.push_back(std::move(mesh));
      meshHash[{i, j}] = m_meshes.size() - 1;
      j++;
    }
    i++;
  }

  // Create prefab for each node
  for (auto i = 0; auto &node : m_internalData->model.nodes) {
    auto meshId = node.mesh;
    if (meshId < 0) {
      continue;
    }
    for (auto j = 0;
         auto &primitive : m_internalData->model.meshes[meshId].primitives) {
      auto prefab =
          std::make_shared<GLTFPrefab>(&m_metadata, this, meshId, j, i);
      auto meshFilter = prefab->m_prefab->addComponent<MeshFilter>();
      meshFilter->setMesh(m_meshes[meshHash[{size_cast<uint32_t>(meshId), j}]]);
      m_prefabs.push_back(std::move(prefab));
      j++;
    }
    i++;
  }
  iInfo("GLTFAsset: loaded "
        "mesh count: {}",
        m_meshes.size());
  iInfo("GLTFAsset: loaded "
        "prefab count: {}",
        m_prefabs.size());
}

IFRIT_APIDECL GLTFInternalData *GLTFAsset::getInternalData() {
  return m_internalData;
}

IFRIT_APIDECL GLTFAsset::~GLTFAsset() {
  if (m_internalData) {
    delete m_internalData;
  }
}

// Importer
IFRIT_APIDECL void GLTFAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

IFRIT_APIDECL std::vector<std::string>
GLTFAssetImporter::getSupportedExtensionNames() {
  return {".gltf", ".glb"};
}

IFRIT_APIDECL void
GLTFAssetImporter::importAsset(const std::filesystem::path &path,
                               AssetMetadata &metadata) {
  auto asset = std::make_shared<GLTFAsset>(metadata, path);
  m_assetManager->registerAsset(asset);

  iInfo("Imported asset: "
        "[GLTFObject] {}",
        metadata.m_uuid);
}

} // namespace Ifrit::Core