#define IFRIT_DLL
#include <common/math/LinalgOps.h>
#include <memory>
#include <random>
#include <display/include/presentation/backend/OpenGLBackend.h>
#include <display/include/presentation/window/GLFWWindowProvider.h>
#include <chrono>
#include <fstream>
#include <vkrenderer/include/engine/vkrenderer/Binding.h>
#include <vkrenderer/include/engine/vkrenderer/Command.h>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vkrenderer/include/engine/vkrenderer/RenderGraph.h>
#include <vkrenderer/include/engine/vkrenderer/Shader.h>
#include <vkrenderer/include/engine/vkrenderer/StagedMemoryResource.h>
#include <vkrenderer/include/engine/vkrenderer/Swapchain.h>

#include <meshproclib/include/engine/clusterlod/MeshClusterLodProc.h>
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

int demo_vulkanMeshShader() {
  using namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics;
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

  InitializeArguments args;
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

  EngineContext context(args);
  Swapchain swapchain(&context);

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

  using namespace Ifrit::Engine::MeshProcLib::ClusterLod;
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
  Viewport viewport = {0.0f, 0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, 1.0f};
  Scissor scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};

  DescriptorManager bindlessDesc(&context);
  ResourceManager resourceManager(&context);
  CommandExecutor backend(&context, &swapchain, &bindlessDesc,
                          &resourceManager);
  backend.setQueues(true, 1, 1, 1);

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

  auto meshletBuffer = resourceManager.createStorageBufferDevice(
      meshlets.size() * sizeof(meshopt_Meshlet),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto meshletVertexBuffer = resourceManager.createStorageBufferDevice(
      meshlet_vertices.size() * sizeof(unsigned int),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto meshletIndexBuffer = resourceManager.createStorageBufferDevice(
      meshlet_triangles2.size() * sizeof(unsigned int),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto meshletGraphPartitionBuffer = resourceManager.createStorageBufferDevice(
      meshlet_graphPart.size() * sizeof(unsigned int),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto vertexBuffer = resourceManager.createStorageBufferDevice(
      verticesAligned.size() * sizeof(ifloat4),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto ubBuffer =
      resourceManager.createUniformBufferShared(sizeof(UniformBuffer), true);

  // Culling pipeline
  auto indirectDrawBuffer =
      resourceManager.createIndirectMeshDrawBufferDevice(1);
  auto filteredMeshletBuffer = resourceManager.createStorageBufferDevice(
      meshlets.size() * sizeof(uint32_t));
  auto bvhNodeBuffer = resourceManager.createStorageBufferDevice(
      bvhNodes.size() * sizeof(FlattenedBVHNode),VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto clusterGroupBuffer = resourceManager.createStorageBufferDevice(
      clusterGroupData.size() * sizeof(ClusterGroup),VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto consumerCounterBuffer =
      resourceManager.createStorageBufferDevice(sizeof(uint32_t));
  auto producerCounterBuffer =
      resourceManager.createStorageBufferDevice(sizeof(uint32_t));
  auto remainingCounterBuffer = 
      resourceManager.createStorageBufferDevice(sizeof(uint32_t));
  auto productQueueBuffer = resourceManager.createStorageBufferDevice(
      sizeof(uint32_t) * bvhNodes.size());
  auto ubCullBuffer = resourceManager.createUniformBufferShared(
      sizeof(UniformCullBuffer), true);
  auto meshletInClusteBuffer = resourceManager.createStorageBufferDevice(
      meshlet_inClusterGroup.size() * sizeof(uint32_t),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  uniformData.meshletCount = meshlet_count;

  if (true) {
    StagedSingleBuffer stagedMeshletBuffer(&context, meshletBuffer);
    StagedSingleBuffer stagedMeshletVertexBuffer(&context, meshletVertexBuffer);
    StagedSingleBuffer stagedMeshletIndexBuffer(&context, meshletIndexBuffer);
    StagedSingleBuffer stagedVertexBuffer(&context, vertexBuffer);
    // Culling pipeline
    StagedSingleBuffer stagedBvhNodeBuffer(&context, bvhNodeBuffer);
    StagedSingleBuffer stagedClusterGroupBuffer(&context, clusterGroupBuffer);
    StagedSingleBuffer stagedGraphPartitionBuffer(&context,
                                                  meshletGraphPartitionBuffer);
    StagedSingleBuffer stagedMeshletInClusterBuffer(&context,
                                                    meshletInClusteBuffer);


    backend.runImmidiateCommand(
        [&](CommandBuffer *cmd) -> void {
          stagedMeshletBuffer.cmdCopyToDevice(
              cmd, meshlets.data(), meshlets.size() * sizeof(meshopt_Meshlet),
              0);
          stagedMeshletVertexBuffer.cmdCopyToDevice(
              cmd, meshlet_vertices.data(),
              meshlet_vertices.size() * sizeof(unsigned int), 0);
          stagedMeshletIndexBuffer.cmdCopyToDevice(
              cmd, meshlet_triangles2.data(),
              meshlet_triangles2.size() * sizeof(unsigned int), 0);
          stagedVertexBuffer.cmdCopyToDevice(
              cmd, verticesAligned.data(),
              verticesAligned.size() * sizeof(ifloat4), 0);
          stagedBvhNodeBuffer.cmdCopyToDevice(
              cmd, bvhNodes.data(), bvhNodes.size() * sizeof(FlattenedBVHNode),
              0);
          stagedClusterGroupBuffer.cmdCopyToDevice(
              cmd, clusterGroupData.data(),
              clusterGroupData.size() * sizeof(ClusterGroup), 0);
          stagedGraphPartitionBuffer.cmdCopyToDevice(
              cmd, meshlet_graphPart.data(),
              meshlet_graphPart.size() * sizeof(unsigned int), 0);
          stagedMeshletInClusterBuffer.cmdCopyToDevice(
              cmd, meshlet_inClusterGroup.data(),
              meshlet_inClusterGroup.size() * sizeof(uint32_t), 0);
         
        },
        QueueRequirement::Transfer);
  }

  // Shader
  auto msCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.mesh.spv");
  auto fsCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.frag.spv");
  auto csDynLodCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.dynlod.comp.spv");

  ShaderModule msModule(&context, msCode, "main", ShaderStage::Mesh);
  ShaderModule fsModule(&context, fsCode, "main", ShaderStage::Fragment);
  ShaderModule csDynLodModule(&context, csDynLodCode, "main", ShaderStage::Compute);

  // Depth Buffer
  auto depthImage =
      resourceManager.createDepthAttachment(WINDOW_WIDTH, WINDOW_HEIGHT);

  // Render graph
  auto renderGraph = backend.createRenderGraph();
  auto swapchainRes = backend.getSwapchainImageResource();
  auto msBufReg = renderGraph->registerBuffer(meshletBuffer);
  auto msVBufReg = renderGraph->registerBuffer(meshletVertexBuffer);
  auto msIBufReg = renderGraph->registerBuffer(meshletIndexBuffer);
  auto uniformBufReg = renderGraph->registerBuffer(ubBuffer);
  auto vertexBufReg = renderGraph->registerBuffer(vertexBuffer);
  auto swapchainResReg = renderGraph->registerImage(swapchainRes);
  auto depthReg = renderGraph->registerImage(depthImage);
  auto indirectDrawReg = renderGraph->registerBuffer(indirectDrawBuffer);
  auto graphPartitionReg =
      renderGraph->registerBuffer(meshletGraphPartitionBuffer);
  
  auto uniformBufCullReg = renderGraph->registerBuffer(ubCullBuffer);
  auto queueReg = renderGraph->registerBuffer(productQueueBuffer);
  auto producerReg = renderGraph->registerBuffer(producerCounterBuffer);
  auto consumerReg = renderGraph->registerBuffer(consumerCounterBuffer);
  auto remainingReg = renderGraph->registerBuffer(remainingCounterBuffer);

  auto filteredMeshletReg = renderGraph->registerBuffer(filteredMeshletBuffer);
  auto bvhNodeReg = renderGraph->registerBuffer(bvhNodeBuffer);
  auto clusterReg = renderGraph->registerBuffer(clusterGroupBuffer);
  auto meshletInClusterGroupReg =
      renderGraph->registerBuffer(meshletInClusteBuffer);

  // Cull Pass
  auto cullPass = renderGraph->addComputePass();
  cullPass->setComputeShader(&csDynLodModule);
  cullPass->setPassDescriptorLayout(
      {DescriptorType::StorageBuffer, DescriptorType::StorageBuffer,
       DescriptorType::StorageBuffer, DescriptorType::StorageBuffer,
       DescriptorType::StorageBuffer, DescriptorType::UniformBuffer, 
       DescriptorType::UniformBuffer, 
      // Culling 
       DescriptorType::StorageBuffer, DescriptorType::StorageBuffer, 
       DescriptorType::StorageBuffer, DescriptorType::StorageBuffer});
  // Data buffers
  cullPass->addStorageBuffer(indirectDrawReg, 0, ResourceAccessType::Write);
  cullPass->addStorageBuffer(clusterReg, 1, ResourceAccessType::Read);
  cullPass->addStorageBuffer(bvhNodeReg, 2, ResourceAccessType::Read);
  cullPass->addStorageBuffer(filteredMeshletReg, 3, ResourceAccessType::Write);
  cullPass->addStorageBuffer(meshletInClusterGroupReg, 4,
                             ResourceAccessType::Read);
  cullPass->addUniformBuffer(uniformBufCullReg, 5);
  cullPass->addUniformBuffer(uniformBufReg, 6);

  // Consumer-Producer buffers
  cullPass->addStorageBuffer(consumerReg, 7, ResourceAccessType::Write);
  cullPass->addStorageBuffer(producerReg, 8, ResourceAccessType::Write);
  cullPass->addStorageBuffer(queueReg, 9, ResourceAccessType::Write);
  cullPass->addStorageBuffer(remainingReg, 10, ResourceAccessType::Write);
  cullPass->setRecordFunction(
      [&](RenderPassContext *ctx) -> void { ctx->m_cmd->dispatch(1, 1, 1); });

  // Draw Pass
  // TODO: register indirect
  auto msPass = renderGraph->addGraphicsPass();
  msPass->addColorAttachment(swapchainResReg, VK_ATTACHMENT_LOAD_OP_CLEAR,
                             {{0.00f, 0.00f, 0.00f, 1.0f}});
  msPass->addStorageBuffer(msBufReg, 0, ResourceAccessType::Read);
  msPass->addStorageBuffer(msVBufReg, 1, ResourceAccessType::Read);
  msPass->addStorageBuffer(msIBufReg, 2, ResourceAccessType::Read);
  msPass->addStorageBuffer(vertexBufReg, 3, ResourceAccessType::Read);
  msPass->addStorageBuffer(filteredMeshletReg, 4, ResourceAccessType::Read);
  msPass->addStorageBuffer(graphPartitionReg, 5, ResourceAccessType::Read);
  msPass->addUniformBuffer(uniformBufReg, 6);
  msPass->setPassDescriptorLayout(
      {DescriptorType::StorageBuffer, DescriptorType::StorageBuffer,
       DescriptorType::StorageBuffer, DescriptorType::StorageBuffer,
       DescriptorType::StorageBuffer, DescriptorType::StorageBuffer,
       DescriptorType::UniformBuffer});

  msPass->setMeshShader(&msModule);
  msPass->setFragmentShader(&fsModule);
  msPass->setRasterizerTopology(RasterizerTopology::TriangleList);
  msPass->setRenderArea(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
  msPass->setDepthAttachment(depthReg, VK_ATTACHMENT_LOAD_OP_CLEAR,
                             {{1.0f, 0.0f}});
  msPass->setDepthCompareOp(VK_COMPARE_OP_LESS);
  msPass->setDepthWrite(true);
  msPass->setDepthTestEnable(true);

  msPass->setRecordFunction([&](RenderPassContext *ctx) -> void {
    ctx->m_cmd->setScissors({scissor});
    ctx->m_cmd->setViewports({viewport});
    ctx->m_cmd->drawMeshTasksIndirect(indirectDrawBuffer->getBuffer(), 0, 1, 0);
  });

  float timeVal = 0.0f;
  msPass->setExecutionFunction([&](RenderPassContext *ctx) -> void {
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
    buf->copyToBuffer(&uniformData, sizeof(UniformBuffer), 0);
    buf->flush();
    buf->unmap();
    timeVal += 0.0005f;

    uniformCullData.totalBvhNodes = bvhNodes.size();
    uniformCullData.clusterGroupCounts = clusterGroupData.size();
    uniformCullData.totalLods = MAX_LOD;
    auto buf2 = ubCullBuffer->getActiveBuffer();
    buf2->map();
    buf2->copyToBuffer(&uniformCullData, sizeof(UniformCullBuffer), 0);
    buf2->flush();
    buf2->unmap();
  });

  // Main loop
  windowProvider.loop([&](int *repCore) {
    auto startTime = std::chrono::high_resolution_clock::now();
    backend.beginFrame();
    backend.runRenderGraph(renderGraph);
    backend.endFrame();
    auto endTime = std::chrono::high_resolution_clock::now();
    *repCore = std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                                     startTime)
                   .count();
  });
  context.waitIdle();
  return 0;
}


int main() { demo_vulkanMeshShader(); }