
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
#include "ifrit/core/assetmanager/TextureAsset.h"
#include "ifrit/core/material/SyaroDefaultGBufEmitter.h"
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
  std::shared_ptr<GraphicsBackend::Rhi::RhiSampler> defaultSampler;
};

// Mesh class
IFRIT_APIDECL GLTFPrefab::GLTFPrefab(AssetMetadata *metadata, GLTFAsset *asset,
                                     uint32_t meshId, uint32_t primitiveId,
                                     uint32_t nodeId,
                                     const float4x4 &parentTransform)
    : m_asset(asset), m_meshId(meshId), m_primitiveId(primitiveId),
      m_nodeId(nodeId) {
  m_prefab = SceneObject::createPrefab();

  auto transform = m_prefab->getComponent<Transform>();
  auto data = asset->getInternalData();

  auto &gltfNode = data->model.nodes[nodeId];
  if (gltfNode.matrix.size()) {
    iError("GLTFPrefab: matrix is not supported yet");
  }
  float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
  float scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;
  float rotX = 0.0f, rotY = 0.0f, rotZ = 0.0f, rotW = 1.0f;
  if (gltfNode.translation.size()) {
    posX = static_cast<float>(gltfNode.translation[0]);
    posY = static_cast<float>(gltfNode.translation[1]);
    posZ = static_cast<float>(gltfNode.translation[2]);
  }
  if (gltfNode.scale.size()) {
    scaleX = static_cast<float>(gltfNode.scale[0]);
    scaleY = static_cast<float>(gltfNode.scale[1]);
    scaleZ = static_cast<float>(gltfNode.scale[2]);
  }
  if (gltfNode.rotation.size()) {
    rotX = static_cast<float>(gltfNode.rotation[0]);
    rotY = static_cast<float>(gltfNode.rotation[1]);
    rotZ = static_cast<float>(gltfNode.rotation[2]);
    rotW = static_cast<float>(gltfNode.rotation[3]);
    ifloat3 euler = Math::quaternionToEuler({rotX, rotY, rotZ, rotW});
    rotX = euler.x;
    rotY = euler.y;
    rotZ = euler.z;
  }

  auto localTransform = Math::getTransformMat(
      {scaleX, scaleY, scaleZ}, {posX, posY, posZ}, {rotX, rotY, rotZ});
  auto combinedTransform = Math::matmul(parentTransform, localTransform);

  ifloat3 newScale, newTranslation, newRotation;
  Math::recoverTransformInfo(combinedTransform, newScale, newTranslation,
                             newRotation);
  transform->setPosition(newTranslation);
  transform->setRotation(newRotation);
  transform->setScale(newScale);
}

IFRIT_APIDECL std::shared_ptr<MeshData> GLTFMesh::loadMesh() {
  if (m_loaded) {
    return m_selfData;
  }
  // Start loading mesh
  m_selfData = std::make_shared<MeshData>();
  m_selfData->identifier = m_asset->getMetadata().m_uuid + "_" +
                           std::to_string(m_meshId) + "_" +
                           std::to_string(m_primitiveId);

  auto &data = m_asset->getInternalData()->model;
  auto &mesh = data.meshes[m_meshId];
  auto &primitive = mesh.primitives[m_primitiveId];

  auto &accessorPos = data.accessors[primitive.attributes["POSITION"]];
  auto &accessorNormal = data.accessors[primitive.attributes["NORMAL"]];
  auto &accessorTexcoord = data.accessors[primitive.attributes["TEXCOORD_0"]];
  auto &accessorTangent = data.accessors[primitive.attributes["TANGENT"]];
  auto &accessorIndices = data.accessors[primitive.indices];

  bool tangent3 = false;

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
  iAssertion(accessorTangent.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT,
             "GLTFMesh: component type not supported");
  if (accessorTangent.type == TINYGLTF_TYPE_VEC3) {
    tangent3 = true;
  } else {
    iAssertion(accessorTangent.type == TINYGLTF_TYPE_VEC4,
               "GLTFMesh: type not supported");
  }
  iAssertion(accessorIndices.type == TINYGLTF_TYPE_SCALAR,
             "GLTFMesh: type not supported");

  auto &posBufferView = data.bufferViews[accessorPos.bufferView];
  auto &normalBufferView = data.bufferViews[accessorNormal.bufferView];
  auto &texcoordBufferView = data.bufferViews[accessorTexcoord.bufferView];
  auto &indicesBufferView = data.bufferViews[accessorIndices.bufferView];
  auto &tangentBufferView = data.bufferViews[accessorTangent.bufferView];

  auto &posBuffer = data.buffers[posBufferView.buffer];
  auto &normalBuffer = data.buffers[normalBufferView.buffer];
  auto &texcoordBuffer = data.buffers[texcoordBufferView.buffer];
  auto &indicesBuffer = data.buffers[indicesBufferView.buffer];
  auto &tangentBuffer = data.buffers[tangentBufferView.buffer];

  auto posData = reinterpret_cast<float *>(posBuffer.data.data() +
                                           posBufferView.byteOffset +
                                           accessorPos.byteOffset);
  auto normalData = reinterpret_cast<float *>(normalBuffer.data.data() +
                                              normalBufferView.byteOffset +
                                              accessorNormal.byteOffset);
  auto texcoordData = reinterpret_cast<float *>(texcoordBuffer.data.data() +
                                                texcoordBufferView.byteOffset +
                                                accessorTexcoord.byteOffset);
  auto tangentData = reinterpret_cast<float *>(tangentBuffer.data.data() +
                                               tangentBufferView.byteOffset +
                                               accessorTangent.byteOffset);
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
  auto tangentStride = tangent3 ? 3 : 4;
  auto tangentDataSize =
      size_cast<uint32_t>(accessorTangent.count * tangentStride);

  m_selfData->m_vertices.resize(accessorPos.count);
  m_selfData->m_verticesAligned.resize(accessorPos.count);
  m_selfData->m_normals.resize(accessorNormal.count);
  m_selfData->m_normalsAligned.resize(accessorNormal.count);
  m_selfData->m_uvs.resize(accessorTexcoord.count);
  m_selfData->m_indices.resize(accessorIndices.count);
  m_selfData->m_tangents.resize(accessorTangent.count);

  for (auto i = 0; i < accessorPos.count; i++) {
    m_selfData->m_vertices[i] = {posData[i * 3], posData[i * 3 + 1],
                                 posData[i * 3 + 2]};
    m_selfData->m_verticesAligned[i] = {posData[i * 3], posData[i * 3 + 1],
                                        posData[i * 3 + 2], 1.0f};
  }
  for (auto i = 0; i < accessorNormal.count; i++) {
    m_selfData->m_normals[i] = {normalData[i * 3], normalData[i * 3 + 1],
                                -normalData[i * 3 + 2]};
    m_selfData->m_normalsAligned[i] = {normalData[i * 3], normalData[i * 3 + 1],
                                       normalData[i * 3 + 2], 0.0f};
  }
  for (auto i = 0; i < accessorTexcoord.count; i++) {
    m_selfData->m_uvs[i] = {texcoordData[i * 2], texcoordData[i * 2 + 1]};
  }
  for (auto i = 0; i < accessorTangent.count; i++) {
    m_selfData->m_tangents[i] = {
        tangentData[i * tangentStride], tangentData[i * tangentStride + 1],
        tangentData[i * tangentStride + 2],
        tangent3 ? 1.0f : tangentData[i * tangentStride + 3]};
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
  this->createMeshLodHierarchy(m_selfData, m_cachePath);
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

IFRIT_APIDECL void GLTFAsset::loadGLTF(AssetManager *m_manager) {
  m_internalData = new GLTFInternalData();
  auto rhi = m_manager->getApplication()->getRhiLayer();
  if (m_internalData->defaultSampler == nullptr) {
    m_internalData->defaultSampler = rhi->createTrivialBilinearSampler(true);
  }
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
  auto cachePath = m_manager->getApplication()->getCacheDirectory();
  for (auto i = 0; auto &mesh : m_internalData->model.meshes) {
    for (auto j = 0; auto &primitive : mesh.primitives) {
      auto mesh =
          std::make_shared<GLTFMesh>(&m_metadata, this, i, j, ~0u, cachePath);
      m_meshes.push_back(std::move(mesh));
      meshHash[{i, j}] = m_meshes.size() - 1;
      j++;
    }
    i++;
  }

  // Create prefab for each node
  // get gltf path
  auto gltfDir = m_path.parent_path();

  auto rawGLTFData = m_internalData->model;

  std::function<void(int, float4x4)> traverseNode =
      [&](int nodeId, float4x4 parentTransform) {
        auto &node = rawGLTFData.nodes[nodeId];
        auto translationX = 0.0f, translationY = 0.0f, translationZ = 0.0f;
        auto scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;
        auto rotX = 0.0f, rotY = 0.0f, rotZ = 0.0f, rotW = 1.0f;
        if (node.translation.size()) {
          translationX = static_cast<float>(node.translation[0]);
          translationY = static_cast<float>(node.translation[1]);
          translationZ = static_cast<float>(node.translation[2]);
        }
        if (node.scale.size()) {
          scaleX = static_cast<float>(node.scale[0]);
          scaleY = static_cast<float>(node.scale[1]);
          scaleZ = static_cast<float>(node.scale[2]);
        }
        if (node.rotation.size()) {
          rotX = static_cast<float>(node.rotation[0]);
          rotY = static_cast<float>(node.rotation[1]);
          rotZ = static_cast<float>(node.rotation[2]);
          rotW = static_cast<float>(node.rotation[3]);
          if (node.rotation.size() == 4) {
            ifloat3 euler = Math::quaternionToEuler({rotX, rotY, rotZ, rotW});
            rotX = euler.x;
            rotY = euler.y;
            rotZ = euler.z;
          }
        }
        auto childTransform = Math::getTransformMat(
            {scaleX, scaleY, scaleZ},
            {translationX, translationY, translationZ}, {rotX, rotY, rotZ});

        // Check children
        if (node.children.size()) {
          for (auto &childId : node.children) {
            traverseNode(childId,
                         Math::matmul(parentTransform, childTransform));
          }
        }
        // if mesh is present
        if (node.mesh >= 0) {
          for (auto j = 0;
               auto &primitive : rawGLTFData.meshes[node.mesh].primitives) {
            auto prefab = std::make_shared<GLTFPrefab>(
                &m_metadata, this, node.mesh, j, nodeId, parentTransform);
            auto meshFilter = prefab->m_prefab->addComponent<MeshFilter>();
            auto mesh = m_meshes[meshHash[{size_cast<uint32_t>(node.mesh), j}]];
            meshFilter->setMesh(mesh);

            auto &gltfMaterial = rawGLTFData.materials[primitive.material];
            auto &normalData = gltfMaterial.normalTexture;
            auto &pbrData = gltfMaterial.pbrMetallicRoughness;
            auto baseColorIndex = pbrData.baseColorTexture.index;
            auto normalTexIndex = normalData.index;
            auto baseColorURI = rawGLTFData.images[baseColorIndex].uri;
            auto normalTexURI = rawGLTFData.images[normalTexIndex].uri;

            auto texPathBase = gltfDir / baseColorURI;
            auto texBase = m_manager->requestAsset<Asset>(texPathBase);
            auto texCastedBase =
                std::dynamic_pointer_cast<TextureAsset>(texBase);
            if (!texCastedBase) {
              iError("GLTFAsset: invalid texture asset");
              std::abort();
            }

            auto texPathNormal = gltfDir / normalTexURI;
            auto texNormal = m_manager->requestAsset<Asset>(texPathNormal);
            auto texCastedNormal =
                std::dynamic_pointer_cast<TextureAsset>(texNormal);
            if (!texCastedNormal) {
              iError("GLTFAsset: invalid texture asset");
              std::abort();
            }

            auto meshRenderer = prefab->m_prefab->addComponent<MeshRenderer>();
            auto material = std::make_shared<SyaroDefaultGBufEmitter>(
                m_manager->getApplication());
            auto albedoId = rhi->registerCombinedImageSampler(
                texCastedBase->getTexture().get(),
                m_internalData->defaultSampler.get());
            auto normalId = rhi->registerCombinedImageSampler(
                texCastedNormal->getTexture().get(),
                m_internalData->defaultSampler.get());

            material->setAlbedoId(albedoId->getActiveId());
            material->setNormalMapId(normalId->getActiveId());
            material->buildMaterial();
            meshRenderer->setMaterial(material);

            m_prefabs.push_back(std::move(prefab));
            j++;
          }
        }
      };

  // Get nodes from scenes
  for (auto &scene : rawGLTFData.scenes) {
    for (auto &nodeId : scene.nodes) {
      traverseNode(nodeId, Math::identity());
    }
  }
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
  auto asset = std::make_shared<GLTFAsset>(metadata, path, m_assetManager);
  m_assetManager->registerAsset(asset);

  // iInfo("Imported asset: [GLTFObject] {}", metadata.m_uuid);
}

} // namespace Ifrit::Core