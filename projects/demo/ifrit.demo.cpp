#ifndef IFRIT_DLL
#define IFRIT_DLL
#endif
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/core/Core.h"
#include "ifrit/display/presentation/backend/OpenGLBackend.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Backend.h"
#include <chrono>
#include <fstream>
#include <memory>
#include <random>

#include "ifrit/meshproc/engine/clusterlod/MeshClusterLodProc.h"
#include <meshoptimizer/src/meshoptimizer.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define WINDOW_WIDTH 1980
#define WINDOW_HEIGHT 1080

void loadWaveFrontObject(const char *path, std::vector<ifloat3> &vertices,
                         std::vector<ifloat3> &normals,
                         std::vector<ifloat2> &uvs,
                         std::vector<uint32_t> &indices) {

  // This section is auto-generated from Copilot
  std::ifstream file(path);
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "v") {
      ifloat3 vertex;
      iss >> vertex.x >> vertex.y >> vertex.z;
      vertices.push_back(vertex);
    } else if (type == "vn") {
      ifloat3 normal;
      iss >> normal.x >> normal.y >> normal.z;
      normals.push_back(normal);
    } else if (type == "vt") {
      ifloat2 uv;
      iss >> uv.x >> uv.y;
      uvs.push_back(uv);
    } else if (type == "f") {
      std::string vertex;
      for (int i = 0; i < 3; i++) {
        iss >> vertex;
        std::istringstream vss(vertex);
        std::string index;
        for (int j = 0; j < 3; j++) {
          std::getline(vss, index, '/');
          if (index.size() != 0) {
            indices.push_back(std::stoi(index) - 1);
          } else {
            indices.push_back(0);
          }
        }
      }
    }
  }
}

std::vector<char> readShaderFile(std::string filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }
  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();
  return buffer;
}

// Glfw key function here
float movLeft = 0, movRight = 0, movTop = 0, movBottom = 0, movFar = 0,
      movNear = 0;

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  auto scale = 0.3f;
  if (key == GLFW_KEY_A && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    movLeft += scale;
  }
  if (key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    movRight += scale;
  }
  if (key == GLFW_KEY_W && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    movTop += scale;
  }
  if (key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    movBottom += scale;
  }
  if (key == GLFW_KEY_E && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    movFar += scale;
  }
  if (key == GLFW_KEY_F && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
    movNear += scale;
  }
}

static void error_callback(int error, const char *description) {
  fprintf(stderr, "Error: %s\n", description);
  std::abort();
}

int demo_meshShader() {
  using namespace Ifrit::GraphicsBackend::Rhi;
  using namespace Ifrit::GraphicsBackend::VulkanGraphics;
  using namespace Ifrit::Presentation::Window;
  using namespace Ifrit::Math;

  glfwSetErrorCallback(error_callback);
  glfwInit();
  GLFWWindowProviderInitArgs glfwArgs;
  glfwArgs.vulkanMode = true;
  GLFWWindowProvider windowProvider(glfwArgs);
  windowProvider.callGlfwInit();
  windowProvider.setup(WINDOW_WIDTH, WINDOW_HEIGHT);
  windowProvider.setTitle("Ifrit-v2 (Vulkan Backend)");

  // TODO: window class
  glfwSetKeyCallback((GLFWwindow *)windowProvider.getGLFWWindow(),
                     key_callback);

  RhiInitializeArguments args;
  auto extensionGetter = [&windowProvider](uint32_t *count) -> const char ** {
    return windowProvider.getVkRequiredInstanceExtensions(count);
  };
  auto fbSize = windowProvider.getFramebufferSize();
  args.m_extensionGetter = extensionGetter;
  args.m_win32.m_hInstance = GetModuleHandle(NULL);
  args.m_win32.m_hWnd = (HWND)windowProvider.getWindowObject();
  args.m_surfaceWidth = fbSize.first;
  args.m_surfaceHeight = fbSize.second;
  args.m_expectedSwapchainImageCount = 2;
  args.m_expectedGraphicsQueueCount = 1;
  args.m_expectedComputeQueueCount = 1;
  args.m_expectedTransferQueueCount = 1;

  RhiVulkanBackendBuilder builder;
  auto rt = builder.createBackend(args);

  // load meshlet
  std::vector<ifloat3> vertices;
  std::vector<ifloat4> verticesAligned;
  std::vector<ifloat3> normals;
  std::vector<ifloat2> uvs;
  std::vector<uint32_t> indicesRaw;
  std::vector<uint32_t> indices;
  loadWaveFrontObject(IFRIT_DEMO_ASSET_PATH "/bunny.obj", vertices,
                      normals, //"C:/WR/hk3.obj"
                      uvs, indicesRaw);

  indices.resize(indicesRaw.size() / 3);
  for (int i = 0; i < indicesRaw.size(); i += 3) {
    indices[i / 3] = indicesRaw[i];
  }
  verticesAligned.resize(vertices.size());
  for (int i = 0; i < vertices.size(); i++) {
    verticesAligned[i] =
        ifloat4(vertices[i].x, vertices[i].y, vertices[i].z, 1.0);
  }

  const size_t max_vertices = 64;
  const size_t max_triangles = 124;
  const float cone_weight = 0.0f;

  using namespace Ifrit::MeshProcLib::ClusterLod;
  MeshClusterLodProc meshProc;
  MeshDescriptor meshDesc;
  meshDesc.indexCount = indices.size();
  meshDesc.indexData = reinterpret_cast<char *>(indices.data());
  meshDesc.positionOffset = 0;
  meshDesc.vertexCount = vertices.size();
  meshDesc.vertexData = reinterpret_cast<char *>(vertices.data());
  meshDesc.vertexStride = sizeof(ifloat3);

  constexpr int MAX_LOD = 4;
  auto chosenLod = MAX_LOD - 1;
  CombinedClusterLodBuffer meshletData;
  std::vector<ClusterGroup> clusterGroupData;
  std::vector<FlattenedBVHNode> bvhNodes;
  meshProc.clusterLodHierachy(meshDesc, meshletData, clusterGroupData, bvhNodes,
                              MAX_LOD);

  auto meshlet_triangles = meshletData.meshletTriangles;
  auto meshlets = meshletData.meshletsRaw;
  auto meshlet_vertices = meshletData.meshletVertices;
  auto meshlet_count = meshlets.size();
  auto meshlet_cull = meshletData.meshletCull;
  auto meshlet_graphPart = meshletData.graphPartition;
  auto meshlet_inClusterGroup = meshletData.meshletsInClusterGroups;

  std::vector<unsigned int> meshlet_triangles2(meshlet_triangles.size());
  for (int i = 0; i < meshlet_triangles.size(); i++) {
    meshlet_triangles2[i] = meshlet_triangles[i];
  }

  printf("Meshlet count: %lld\n", meshlet_count);
  printf("Meshlet Vertices: %lld\n", meshlet_vertices.size());
  printf("Meshlet Triangles: %lld\n", meshlet_triangles.size());
  printf("Vertex Buffer %lld\n", vertices.size());

  printf("Total BVH Nodes:%lld\n", bvhNodes.size());
  printf("Total Cluster Groups:%lld\n", clusterGroupData.size());
  printf("Total Cluster Groups Raw:%lld\n", meshletData.clusterGroups.size());

  // Create ssbo
  RhiViewport viewport = {0.0f, 0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, 1.0f};
  RhiScissor scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};

  struct UniformBuffer {

    float4x4 mvp;
    float4x4 mv;
    ifloat4 cameraPos;
    uint32_t meshletCount;
    float fov;
  } uniformData;

  struct UniformCullBuffer {
    uint32_t clusterGroupCounts;
    uint32_t totalBvhNodes;
    uint32_t totalLods;
  } uniformCullData;

  auto meshletBuffer =
      rt->createStorageBufferDevice(meshlets.size() * sizeof(meshopt_Meshlet),
                                    RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto meshletVertexBuffer = rt->createStorageBufferDevice(
      meshlet_vertices.size() * sizeof(unsigned int),
      RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto meshletIndexBuffer = rt->createStorageBufferDevice(
      meshlet_triangles2.size() * sizeof(unsigned int),
      RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto meshletGraphPartitionBuffer = rt->createStorageBufferDevice(
      meshlet_graphPart.size() * sizeof(unsigned int),
      RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto vertexBuffer =
      rt->createStorageBufferDevice(verticesAligned.size() * sizeof(ifloat4),
                                    RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto ubBuffer = rt->createUniformBufferShared(sizeof(UniformBuffer), true, 0);

  // Culling pipeline
  auto indirectDrawBuffer = rt->createIndirectMeshDrawBufferDevice(1);
  auto filteredMeshletBuffer =
      rt->createStorageBufferDevice(meshlets.size() * sizeof(uint32_t), 0);
  auto bvhNodeBuffer =
      rt->createStorageBufferDevice(bvhNodes.size() * sizeof(FlattenedBVHNode),
                                    RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto clusterGroupBuffer = rt->createStorageBufferDevice(
      clusterGroupData.size() * sizeof(ClusterGroup),
      RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto consumerCounterBuffer =
      rt->createStorageBufferDevice(sizeof(uint32_t), 0);
  auto producerCounterBuffer =
      rt->createStorageBufferDevice(sizeof(uint32_t), 0);
  auto remainingCounterBuffer =
      rt->createStorageBufferDevice(sizeof(uint32_t), 0);
  auto productQueueBuffer =
      rt->createStorageBufferDevice(sizeof(uint32_t) * bvhNodes.size(), 0);
  auto ubCullBuffer =
      rt->createUniformBufferShared(sizeof(UniformCullBuffer), true, 0);
  auto meshletInClusteBuffer = rt->createStorageBufferDevice(
      meshlet_inClusterGroup.size() * sizeof(uint32_t),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  uniformData.meshletCount = meshlet_count;

  if (true) {
    auto stagedMeshletBuffer = rt->createStagedSingleBuffer(meshletBuffer);
    auto stagedMeshletVertexBuffer =
        rt->createStagedSingleBuffer(meshletVertexBuffer);
    auto stagedMeshletIndexBuffer =
        rt->createStagedSingleBuffer(meshletIndexBuffer);
    auto stagedVertexBuffer = rt->createStagedSingleBuffer(vertexBuffer);

    // Culling pipeline
    auto stagedBvhNodeBuffer = rt->createStagedSingleBuffer(bvhNodeBuffer);
    auto stagedClusterGroupBuffer =
        rt->createStagedSingleBuffer(clusterGroupBuffer);
    auto stagedGraphPartitionBuffer =
        rt->createStagedSingleBuffer(meshletGraphPartitionBuffer);
    auto stagedMeshletInClusterBuffer =
        rt->createStagedSingleBuffer(meshletInClusteBuffer);

    auto tq = rt->getQueue(RHI_QUEUE_TRANSFER_BIT);

    tq->runSyncCommand([&](const RhiCommandBuffer *cmd) -> void {
      stagedMeshletBuffer->cmdCopyToDevice(
          cmd, meshlets.data(), meshlets.size() * sizeof(meshopt_Meshlet), 0);
      stagedMeshletVertexBuffer->cmdCopyToDevice(
          cmd, meshlet_vertices.data(),
          meshlet_vertices.size() * sizeof(unsigned int), 0);
      stagedMeshletIndexBuffer->cmdCopyToDevice(
          cmd, meshlet_triangles2.data(),
          meshlet_triangles2.size() * sizeof(unsigned int), 0);
      stagedVertexBuffer->cmdCopyToDevice(
          cmd, verticesAligned.data(), verticesAligned.size() * sizeof(ifloat4),
          0);
      stagedBvhNodeBuffer->cmdCopyToDevice(
          cmd, bvhNodes.data(), bvhNodes.size() * sizeof(FlattenedBVHNode), 0);
      stagedClusterGroupBuffer->cmdCopyToDevice(
          cmd, clusterGroupData.data(),
          clusterGroupData.size() * sizeof(ClusterGroup), 0);
      stagedGraphPartitionBuffer->cmdCopyToDevice(
          cmd, meshlet_graphPart.data(),
          meshlet_graphPart.size() * sizeof(unsigned int), 0);
      stagedMeshletInClusterBuffer->cmdCopyToDevice(
          cmd, meshlet_inClusterGroup.data(),
          meshlet_inClusterGroup.size() * sizeof(uint32_t), 0);
    });
  }

  // Shader
  auto msCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.mesh.spv");
  auto fsCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.frag.spv");
  auto csDynLodCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.dynlod.comp.spv");

  auto msModule = rt->createShader(msCode, "main", RhiShaderStage::Mesh);
  auto fsModule = rt->createShader(fsCode, "main", RhiShaderStage::Fragment);
  auto csDynLodModule =
      rt->createShader(csDynLodCode, "main", RhiShaderStage::Compute);

  // Depth Buffer
  auto depthImage = rt->createDepthRenderTexture(WINDOW_WIDTH, WINDOW_HEIGHT);

  // Cull Pass
  auto cullPass = rt->createComputePass();

  cullPass->setComputeShader(csDynLodModule);
  cullPass->setShaderBindingLayout(
      {RhiDescriptorType::StorageBuffer, RhiDescriptorType::StorageBuffer,
       RhiDescriptorType::StorageBuffer, RhiDescriptorType::StorageBuffer,
       RhiDescriptorType::StorageBuffer, RhiDescriptorType::UniformBuffer,
       RhiDescriptorType::UniformBuffer,
       // Culling
       RhiDescriptorType::StorageBuffer, RhiDescriptorType::StorageBuffer,
       RhiDescriptorType::StorageBuffer, RhiDescriptorType::StorageBuffer});

  // Data buffers
  cullPass->addShaderStorageBuffer(indirectDrawBuffer, 0,
                                   RhiResourceAccessType::Write);
  cullPass->addShaderStorageBuffer(clusterGroupBuffer, 1,
                                   RhiResourceAccessType::Read);
  cullPass->addShaderStorageBuffer(bvhNodeBuffer, 2,
                                   RhiResourceAccessType::Read);
  cullPass->addShaderStorageBuffer(filteredMeshletBuffer, 3,
                                   RhiResourceAccessType::Write);
  cullPass->addShaderStorageBuffer(meshletInClusteBuffer, 4,
                                   RhiResourceAccessType::Read);
  cullPass->addUniformBuffer(ubCullBuffer, 5);
  cullPass->addUniformBuffer(ubBuffer, 6);

  // Consumer-Producer buffers
  cullPass->addShaderStorageBuffer(consumerCounterBuffer, 7,
                                   RhiResourceAccessType::Write);
  cullPass->addShaderStorageBuffer(producerCounterBuffer, 8,
                                   RhiResourceAccessType::Write);
  cullPass->addShaderStorageBuffer(productQueueBuffer, 9,
                                   RhiResourceAccessType::Write);
  cullPass->addShaderStorageBuffer(remainingCounterBuffer, 10,
                                   RhiResourceAccessType::Write);
  cullPass->setRecordFunction([&](RhiRenderPassContext *ctx) -> void {
    ctx->m_cmd->dispatch(1, 1, 1);
  });

  // Draw Pass
  // TODO: register indirect
  auto msPass = rt->createGraphicsPass();
  auto swapchainImg = rt->getSwapchainImage();
  msPass->addColorAttachment(swapchainImg, RhiRenderTargetLoadOp::Clear,
                             {{0.00f, 0.00f, 0.00f, 1.0f}});
  msPass->addShaderStorageBuffer(meshletBuffer, 0, RhiResourceAccessType::Read);
  msPass->addShaderStorageBuffer(meshletVertexBuffer, 1,
                                 RhiResourceAccessType::Read);
  msPass->addShaderStorageBuffer(meshletIndexBuffer, 2,
                                 RhiResourceAccessType::Read);
  msPass->addShaderStorageBuffer(vertexBuffer, 3, RhiResourceAccessType::Read);
  msPass->addShaderStorageBuffer(filteredMeshletBuffer, 4,
                                 RhiResourceAccessType::Read);
  msPass->addShaderStorageBuffer(meshletGraphPartitionBuffer, 5,
                                 RhiResourceAccessType::Read);
  msPass->addUniformBuffer(ubBuffer, 6);
  msPass->setShaderBindingLayout(
      {RhiDescriptorType::StorageBuffer, RhiDescriptorType::StorageBuffer,
       RhiDescriptorType::StorageBuffer, RhiDescriptorType::StorageBuffer,
       RhiDescriptorType::StorageBuffer, RhiDescriptorType::StorageBuffer,
       RhiDescriptorType::UniformBuffer});

  msPass->setMeshShader(msModule);
  msPass->setPixelShader(fsModule);
  msPass->setRasterizerTopology(RhiRasterizerTopology::TriangleList);
  msPass->setRenderArea(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
  msPass->setDepthAttachment(depthImage, RhiRenderTargetLoadOp::Clear,
                             {{}, 1.0f});

  msPass->setDepthCompareOp(RhiCompareOp::Less);
  msPass->setDepthWrite(true);
  msPass->setDepthTestEnable(true);

  msPass->setRecordFunction([&](RhiRenderPassContext *ctx) -> void {
    ctx->m_cmd->setScissors({scissor});
    ctx->m_cmd->setViewports({viewport});
    ctx->m_cmd->drawMeshTasksIndirect(indirectDrawBuffer, 0, 1, 0);
  });
  msPass->setRecordFunctionPostRenderPass(
      [&](RhiRenderPassContext *ctx) -> void {
        ctx->m_cmd->imageBarrier(swapchainImg, RhiResourceState::RenderTarget,
                                 RhiResourceState::Present);
      });

  float timeVal = 0.0f;
  msPass->setExecutionFunction([&](RhiRenderPassContext *ctx) -> void {
    float z = 0.25 + (movFar - movNear) * 0.02; // + 0.5*sin(timeVal);
    float x = 0 + (movRight - movLeft) * 0.02;
    float y = 0.1 + (movTop - movBottom) * 0.02;

    ifloat4 camPos = ifloat4(x, y, z, 1.0);
    float4x4 view = (lookAt({camPos.x, camPos.y, z},
                            {camPos.x, camPos.y, z - 1}, {0, 1, 0}));
    float4x4 proj = (perspectiveNegateY(
        60 * 3.14159 / 180, 1.0 * WINDOW_WIDTH / WINDOW_HEIGHT, 0.01, 3000));
    auto mvp = transpose(matmul(proj, view));
    auto mv = transpose(view);
    uniformData.mvp = mvp;
    uniformData.mv = mv;
    uniformData.fov = 60 * 3.14159 / 180;
    uniformData.cameraPos = camPos;
    auto buf = ubBuffer->getActiveBuffer();
    buf->map();
    buf->writeBuffer(&uniformData, sizeof(UniformBuffer), 0);
    buf->flush();
    buf->unmap();
    timeVal += 0.0005f;

    uniformCullData.totalBvhNodes = bvhNodes.size();
    uniformCullData.clusterGroupCounts = clusterGroupData.size();
    uniformCullData.totalLods = MAX_LOD;
    auto buf2 = ubCullBuffer->getActiveBuffer();
    buf2->map();
    buf2->writeBuffer(&uniformCullData, sizeof(UniformCullBuffer), 0);
    buf2->flush();
    buf2->unmap();
  });

  // Main loop
  auto compQueue = rt->getQueue(RHI_QUEUE_COMPUTE_BIT);
  auto drawQueue = rt->getQueue(RHI_QUEUE_GRAPHICS_BIT);
  windowProvider.loop([&](int *repCore) {
    rt->beginFrame();
    auto sFrameReady = rt->getSwapchainFrameReadyEventHandler();
    auto sRenderComplete = rt->getSwapchainRenderDoneEventHandler();
    auto sCompEnd = compQueue->runAsyncCommand(
        [&](const RhiCommandBuffer *cmd) { cullPass->run(cmd, 0); },
        {sFrameReady.get()}, {});
    auto sDrawEnd = drawQueue->runAsyncCommand(
        [&](const RhiCommandBuffer *cmd) { msPass->run(cmd, 0); },
        {sCompEnd.get()}, {sRenderComplete.get()});

    rt->endFrame();
  });
  rt->waitDeviceIdle();
  return 0;
}

int demo_core() { 
  using namespace Ifrit::Core;
  AssetManager assetManager(IFRIT_DEMO_ASSET_PATH);
  return 0;
}

int main() { demo_core(); }