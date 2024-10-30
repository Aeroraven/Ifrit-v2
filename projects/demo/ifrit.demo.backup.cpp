#define IFRIT_DLL
#include "ifrit/common/math/LinalgOps.h"
#include <memory>
#include <random>

#include <softrenderer/include/engine/tileraster/TileRasterRenderer.h>

#include <display/include/presentation/backend/OpenGLBackend.h>
#include <display/include/presentation/window/GLFWWindowProvider.h>

#include "ifrit/vkgraphics/engine/vkrenderer/Binding.h>
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h>
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h>
#include "ifrit/vkgraphics/engine/vkrenderer/RenderGraph.h>
#include "ifrit/vkgraphics/engine/vkrenderer/Shader.h>
#include "ifrit/vkgraphics/engine/vkrenderer/StagedMemoryResource.h>
#include "ifrit/vkgraphics/engine/vkrenderer/Swapchain.h>
#include <chrono>
#include <fstream>

#include <meshproclib/include/engine/clusterlod/MeshClusterLodProc.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <meshoptimizer/src/meshoptimizer.h>

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

class DemoVS : public Ifrit::Engine::SoftRenderer::VertexShader {
public:
  virtual void execute(const void *const *input, ifloat4 *outPos,
                       ifloat4 *const *outVaryings) {
    const float *inPos = (const float *)input[0];
    outPos->x = inPos[0];
    outPos->y = inPos[1];
    outPos->z = inPos[2];
    outPos->w = 1.0f;
  }
};

class DemoFS : public Ifrit::Engine::SoftRenderer::FragmentShader {
public:
  virtual void execute(const void *varyings, void *colorOutput,
                       float *fragmentDepth) {
    // fill red color
    float *outColor = (float *)colorOutput;
    outColor[0] = 0.2f;
    outColor[1] = 0.0f;
    outColor[2] = 0.0f;
    outColor[3] = 1.0f;
  }
};

int main2() {
  using namespace Ifrit::Engine::SoftRenderer::TileRaster;
  using namespace Ifrit::Engine::SoftRenderer::Core::Data;
  using namespace Ifrit::Engine::SoftRenderer::BufferManager;

  std::shared_ptr<TileRasterRenderer> renderer =
      std::make_shared<TileRasterRenderer>();
  renderer->init();
  std::unique_ptr<ImageF32> color =
      std::make_unique<ImageF32>(WINDOW_WIDTH, WINDOW_HEIGHT, 4);
  std::unique_ptr<ImageF32> depth =
      std::make_unique<ImageF32>(WINDOW_WIDTH, WINDOW_HEIGHT, 1);
  FrameBuffer frameBuffer;
  frameBuffer.setColorAttachments({color.get()});
  frameBuffer.setDepthAttachment(*depth.get());
  renderer->bindFrameBuffer(frameBuffer);
  VertexBuffer vertexBuffer;
  vertexBuffer.setLayout({TypeDescriptors.FLOAT4});
  vertexBuffer.setVertexCount(3);
  vertexBuffer.allocateBuffer(3);

  vertexBuffer.setValue<ifloat4>(0, 0, {0.0f, 0.5f, 0.2f, 1.0f});
  vertexBuffer.setValue<ifloat4>(1, 0, {0.5f, -0.5f, 0.2f, 1.0f});
  vertexBuffer.setValue<ifloat4>(2, 0, {-0.5f, -0.5f, 0.2f, 1.0f});
  renderer->bindVertexBuffer(vertexBuffer);

  DemoVS vs;
  renderer->bindVertexShader(vs);
  DemoFS fs;
  renderer->bindFragmentShader(fs);

  std::vector<int> indexBuffer = {2, 1, 0};
  std::shared_ptr<TrivialBufferManager> bufferman =
      std::make_shared<TrivialBufferManager>();
  bufferman->init();
  auto indexBuffer1 =
      bufferman->createBuffer({sizeof(indexBuffer[0]) * indexBuffer.size()});
  bufferman->bufferData(indexBuffer1, indexBuffer.data(), 0,
                        sizeof(indexBuffer[0]) * indexBuffer.size());
  renderer->bindIndexBuffer(indexBuffer1);

  using namespace Ifrit::Display::Backend;
  using namespace Ifrit::Display::Window;
  GLFWWindowProvider windowProvider;
  windowProvider.setup(WINDOW_WIDTH, WINDOW_HEIGHT);
  windowProvider.setTitle("Ifrit");

  OpenGLBackend backend;
  backend.setViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

  windowProvider.loop([&](int *repCore) {
    renderer->drawElements(3, true);
    backend.updateTexture(color->getData(), 4, WINDOW_WIDTH, WINDOW_HEIGHT);
    backend.draw();
  });
  return 0;
}

int demo_vulkanTriangle() {
  using namespace Ifrit::Engine::VkRenderer;
  using namespace Ifrit::Display::Window;
  using namespace Ifrit::Math;

  GLFWWindowProviderInitArgs glfwArgs;
  glfwArgs.vulkanMode = true;
  GLFWWindowProvider windowProvider(glfwArgs);
  windowProvider.setup(WINDOW_WIDTH, WINDOW_HEIGHT);
  windowProvider.setTitle("Ifrit-v2 (Vulkan Backend)");

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

  // Print swapchain queue
  auto vsCode = readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.demo.vert.spv");
  auto fsCode = readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.demo.frag.spv");

  ShaderModule vsModule(&context, vsCode, "main", ShaderStage::Vertex);
  ShaderModule fsModule(&context, fsCode, "main", ShaderStage::Fragment);

  Viewport viewport = {0.0f, 0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, 1.0f};
  Scissor scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};

  std::vector<float> vertexData = {
      -0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, //
      0.5f,  -0.5f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f, 0.0f, //
      0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 1.0f, //
      -0.5f, 0.5f,  0.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, //

      -0.5f, -0.5f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, //
      0.5f,  -0.5f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, //
      0.5f,  0.5f,  -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, //
      -0.5f, 0.5f,  -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f  //
  };
  std::vector<uint32_t> indexData = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};

  // Resource manager
  DescriptorManager bindlessDesc(&context);
  ResourceManager resourceManager(&context);
  CommandExecutor backend(&context, &swapchain, &bindlessDesc,
                          &resourceManager);
  backend.setQueues(true, 1, 1, 1);

  // Image
  int texWidth, texHeight, texChannels;
  stbi_uc *pixels = stbi_load(IFRIT_DEMO_ASSET_PATH "/texture.png", &texWidth,
                              &texHeight, &texChannels, STBI_rgb_alpha);
  uint64_t imageSize = texWidth * texHeight * 4;

  if (!pixels) {
    throw std::runtime_error("failed to load texture image!");
  }
  auto texture = resourceManager.createTexture2DDevice(
      texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // Sampler
  SamplerCreateInfo samplerCI{};
  auto sampler = resourceManager.createSampler(samplerCI);

  // Vertex buffer & index buffer
  VertexBufferDescriptor vbDesc;
  vbDesc.addBinding({0, 1, 2},
                    {VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT,
                     VK_FORMAT_R32G32_SFLOAT},
                    {0, 12, 24}, 32, VertexInputRate::Vertex);

  auto vxBuffer = resourceManager.createVertexBufferDevice(
      vertexData.size() * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto ixBuffer = resourceManager.createIndexBufferDevice(
      indexData.size() * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  // Stage resource
  StagedSingleImage stagedTexture(&context, texture);
  StagedSingleBuffer stagedVertexBuffer(&context, vxBuffer);
  StagedSingleBuffer stagedIndexBuffer(&context, ixBuffer);
  backend.runImmidiateCommand(
      [&](CommandBuffer *cmd) {
        stagedVertexBuffer.cmdCopyToDevice(
            cmd, vertexData.data(), vertexData.size() * sizeof(float), 0);
        stagedIndexBuffer.cmdCopyToDevice(
            cmd, indexData.data(), indexData.size() * sizeof(uint32_t), 0);
        stagedTexture.cmdCopyToDevice(cmd, pixels, VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                      VK_ACCESS_SHADER_READ_BIT);
      },
      QueueRequirement::Transfer);

  // Uniform Buffers
  auto ubBuffer =
      resourceManager.createUniformBufferShared(16 * sizeof(float), true);

  // Depth Buffer
  auto depthImage =
      resourceManager.createDepthAttachment(WINDOW_WIDTH, WINDOW_HEIGHT);

  // Render graph
  auto renderGraph = backend.createRenderGraph();
  auto pass = renderGraph->addGraphicsPass();
  auto swapchainRes = backend.getSwapchainImageResource();

  auto swapchainResReg = renderGraph->registerImage(swapchainRes);
  auto vbReg = renderGraph->registerBuffer(stagedVertexBuffer.getBuffer());
  auto ibReg = renderGraph->registerBuffer(stagedIndexBuffer.getBuffer());
  auto ubReg = renderGraph->registerBuffer(ubBuffer);
  auto imReg = renderGraph->registerImage(texture);
  auto smReg = renderGraph->registerSampler(sampler);
  auto depthReg = renderGraph->registerImage(depthImage);

  pass->addColorAttachment(swapchainResReg, VK_ATTACHMENT_LOAD_OP_CLEAR,
                           {{0.2f, 0.2f, 0.2f, 1.0f}});
  pass->setVertexInput(vbDesc, {vbReg});
  pass->setIndexInput(ibReg, VK_INDEX_TYPE_UINT32);
  pass->setDepthAttachment(depthReg, VK_ATTACHMENT_LOAD_OP_CLEAR,
                           {{1.0f, 0.0f}});
  pass->addUniformBuffer(ubReg, 0);
  pass->addCombinedImageSampler(imReg, smReg, 1);

  pass->setDepthWrite(true);
  pass->setDepthTestEnable(true);
  pass->setDepthCompareOp(VK_COMPARE_OP_LESS);
  pass->setVertexShader(&vsModule);
  pass->setFragmentShader(&fsModule);
  pass->setRenderArea(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

  pass->setPassDescriptorLayout(
      {DescriptorType::UniformBuffer, DescriptorType::CombinedImageSampler});

  // Uniform data
  float4x4 view = (lookAt({0, 0.1, 2.25}, {0, 0.0, 0.0}, {0, 1, 0}));
  float4x4 proj = (perspective(60 * 3.14159 / 180,
                               1.0 * WINDOW_WIDTH / WINDOW_HEIGHT, 0.01, 3000));

  // Pass Record
  pass->setRecordFunction([&](RenderPassContext *ctx) -> void {
    ctx->m_cmd->setScissors({scissor});
    ctx->m_cmd->setViewports({viewport});
    ctx->m_cmd->drawIndexed(12, 1, 0, 0, 0);
  });

  float delta = 0.0f;
  pass->setExecutionFunction([&](RenderPassContext *ctx) -> void {
    float4x4 model = axisAngleRotation(ifloat3(0.0, 1.0, 0.0), delta);
    float4x4 mvp = transpose(matmul(proj, matmul(view, model)));
    delta += 0.001f;
    auto buf = ubBuffer->getActiveBuffer();
    buf->map();
    buf->copyToBuffer(mvp.data, sizeof(mvp), 0);
    buf->flush();
    buf->unmap();
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

int demo_vulkanComputeShader() {
  using namespace Ifrit::Engine::VkRenderer;
  using namespace Ifrit::Display::Window;
  using namespace Ifrit::Math;

  GLFWWindowProviderInitArgs glfwArgs;
  glfwArgs.vulkanMode = true;
  GLFWWindowProvider windowProvider(glfwArgs);
  windowProvider.setup(WINDOW_WIDTH, WINDOW_HEIGHT);
  windowProvider.setTitle("Ifrit-v2 (Vulkan Backend)");

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

  // Print swapchain queue
  auto csCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.particle.comp.spv");
  auto fsCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.particle.frag.spv");
  auto vsCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.particle.vert.spv");

  ShaderModule vsModule(&context, vsCode, "main", ShaderStage::Vertex);
  ShaderModule fsModule(&context, fsCode, "main", ShaderStage::Fragment);
  ShaderModule csModule(&context, csCode, "main", ShaderStage::Compute);

  Viewport viewport = {0.0f, 0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, 1.0f};
  Scissor scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};

  DescriptorManager bindlessDesc(&context);
  ResourceManager resourceManager(&context);
  CommandExecutor backend(&context, &swapchain, &bindlessDesc,
                          &resourceManager);
  backend.setQueues(true, 1, 1, 1);

  // Create ssbo
  constexpr int PARTICLE_COUNT = 1024;
  struct Particle {
    ifloat2 pos;
    ifloat2 vel;
    ifloat4 color;
  };
  auto buffer1 = resourceManager.createStorageBufferDevice(
      PARTICLE_COUNT * sizeof(Particle),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

  auto buffer2 = resourceManager.createStorageBufferDevice(
      PARTICLE_COUNT * sizeof(Particle),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

  StagedSingleBuffer stagedBuf1(&context, buffer1);

  std::default_random_engine rndEngine((unsigned)time(nullptr));
  std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
  std::vector<Particle> particles(PARTICLE_COUNT);
  for (auto &particle : particles) {
    float r = 0.25f * sqrt(rndDist(rndEngine));
    float theta = rndDist(rndEngine) * 2 * 3.14159265358979323846;
    float x = r * cos(theta) * WINDOW_WIDTH / WINDOW_HEIGHT;
    float y = r * sin(theta);
    particle.pos = ifloat2(x, y);
    particle.vel = normalize(ifloat2(x, y)) * 0.025f;
    particle.color = ifloat4(rndDist(rndEngine), rndDist(rndEngine),
                             rndDist(rndEngine), 1.0f);
  }

  backend.runImmidiateCommand(
      [&](CommandBuffer *cmd) -> void {
        stagedBuf1.cmdCopyToDevice(cmd, particles.data(),
                                   PARTICLE_COUNT * sizeof(Particle), 0);
      },
      QueueRequirement::Transfer);

  auto mulBuf1 = resourceManager.createProxyMultiBuffer({buffer1, buffer2});
  auto mulBuf2 = resourceManager.createProxyMultiBuffer({buffer2, buffer1});

  // Compute pass
  auto renderGraph = backend.createRenderGraph();
  auto swapchainRes = backend.getSwapchainImageResource();
  auto ssboIn = renderGraph->registerBuffer(mulBuf1);
  auto ssboOut = renderGraph->registerBuffer(mulBuf2);
  auto swapchainResReg = renderGraph->registerImage(swapchainRes);

  auto compPass = renderGraph->addComputePass();
  compPass->setComputeShader(&csModule);
  compPass->setPassDescriptorLayout(
      {DescriptorType::StorageBuffer, DescriptorType::StorageBuffer});
  compPass->addStorageBuffer(ssboIn, 0, ResourceAccessType::Read);
  compPass->addStorageBuffer(ssboOut, 1, ResourceAccessType::Write);

  compPass->setRecordFunction([&](RenderPassContext *ctx) -> void {
    ctx->m_cmd->dispatch(PARTICLE_COUNT / 32, 1, 1);
  });

  // Draw pass
  // Vertex buffer & index buffer
  VertexBufferDescriptor vbDesc;
  vbDesc.addBinding({0, 1},
                    {VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
                    {0, 16}, 32, VertexInputRate::Vertex);

  auto drawPass = renderGraph->addGraphicsPass();
  drawPass->addColorAttachment(swapchainResReg, VK_ATTACHMENT_LOAD_OP_CLEAR,
                               {{0.2f, 0.2f, 0.2f, 1.0f}});
  drawPass->setVertexInput(vbDesc, {ssboOut});
  drawPass->setVertexShader(&vsModule);
  drawPass->setFragmentShader(&fsModule);
  drawPass->setRasterizerTopology(RasterizerTopology::Point);
  drawPass->setRenderArea(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

  drawPass->setRecordFunction([&](RenderPassContext *ctx) -> void {
    ctx->m_cmd->setScissors({scissor});
    ctx->m_cmd->setViewports({viewport});
    ctx->m_cmd->draw(PARTICLE_COUNT, 1, 0, 0);
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

// Glfw key function here
float movLeft = 0, movRight = 0, movTop = 0, movBottom = 0, movFar = 0,
      movNear = 0;

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  if (key == GLFW_KEY_A && action == GLFW_REPEAT) {
    movLeft += 0.1f;
  }
  if (key == GLFW_KEY_D && action == GLFW_REPEAT) {
    movRight += 0.1f;
  }
  if (key == GLFW_KEY_W && action == GLFW_REPEAT) {
    movTop += 0.1f;
  }
  if (key == GLFW_KEY_S && action == GLFW_REPEAT) {
    movBottom += 0.1f;
  }
  if (key == GLFW_KEY_E && action == GLFW_REPEAT) {
    movFar += 0.1f;
  }
  if (key == GLFW_KEY_F && action == GLFW_REPEAT) {
    movNear += 0.1f;
  }
}

static void error_callback(int error, const char *description) {
  fprintf(stderr, "Error: %s\n", description);
  std::abort();
}

int demo_vulkanMeshShader() {
  using namespace Ifrit::Engine::VkRenderer;
  using namespace Ifrit::Display::Window;
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

  constexpr int MAX_LOD = 10;
  auto chosenLod = MAX_LOD - 1;
  CombinedClusterLodBuffer combinedBuf;
  MeshletClusterInfo clusterInfo;
  meshProc.clusterLodHierachyAll(meshDesc, combinedBuf, clusterInfo, MAX_LOD);

  auto meshlet_triangles = combinedBuf.meshletTriangles;
  auto meshlets = combinedBuf.meshletsRaw;
  auto meshlet_vertices = combinedBuf.meshletVertices;
  auto meshlet_count = meshlets.size();
  auto meshlet_cull = combinedBuf.meshletCull;

  std::vector<unsigned int> meshlet_triangles2(meshlet_triangles.size());
  for (int i = 0; i < meshlet_triangles.size(); i++) {
    meshlet_triangles2[i] = meshlet_triangles[i];
  }

  printf("Meshlet count: %lld\n", meshlet_count);
  printf("Meshlet Vertices: %lld\n", meshlet_vertices.size());
  printf("Meshlet Triangles: %lld\n", meshlet_triangles.size());
  printf("Vertex Buffer %lld\n", vertices.size());

  // Create ssbo
  Viewport viewport = {0.0f, 0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, 1.0f};
  Scissor scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};

  DescriptorManager bindlessDesc(&context);
  ResourceManager resourceManager(&context);
  CommandExecutor backend(&context, &swapchain, &bindlessDesc,
                          &resourceManager);
  backend.setQueues(true, 1, 1, 1);

  struct UniformBuffer {
    ifloat4 cameraPos;
    float4x4 mvp;
    uint32_t meshletCount;
    float fov;
  } uniformData;

  struct UniformCullBuffer {
    uint32_t clusterCounts;
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
  auto vertexBuffer = resourceManager.createStorageBufferDevice(
      verticesAligned.size() * sizeof(ifloat4),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto meshletCullBuffer = resourceManager.createStorageBufferDevice(
      meshlet_cull.size() * sizeof(MeshletCullData),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto ubBuffer =
      resourceManager.createUniformBufferShared(sizeof(UniformBuffer), true);

  // Culling pipeline
  auto indirectDrawBuffer =
      resourceManager.createIndirectMeshDrawBufferDevice(1);
  auto clusterDataBuffer = resourceManager.createStorageBufferDevice(
      clusterInfo.clusterInfo.size() * sizeof(MeshletClusterInfo),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto targetMeshlets = resourceManager.createStorageBufferDevice(
      meshlets.size() * sizeof(uint32_t));
  auto ubCullBuffer = resourceManager.createUniformBufferShared(
      sizeof(UniformCullBuffer), true);

  auto consumerCounterBuffer =
      resourceManager.createStorageBufferDevice(sizeof(uint32_t));
  auto producerCounterBuffer =
      resourceManager.createStorageBufferDevice(sizeof(uint32_t));
  auto remainingCounterBuffer =
      resourceManager.createStorageBufferDevice(sizeof(uint32_t));
  auto productQueueBuffer = resourceManager.createStorageBufferDevice(
      sizeof(uint32_t) * clusterInfo.clusterInfo.size());

  uniformData.meshletCount = meshlet_count;

  if (true) {
    StagedSingleBuffer stagedMeshletBuffer(&context, meshletBuffer);
    StagedSingleBuffer stagedMeshletVertexBuffer(&context, meshletVertexBuffer);
    StagedSingleBuffer stagedMeshletIndexBuffer(&context, meshletIndexBuffer);
    StagedSingleBuffer stagedVertexBuffer(&context, vertexBuffer);
    StagedSingleBuffer stagedMeshletCullBuffer(&context, meshletCullBuffer);
    StagedSingleBuffer stagedClusterDataBuffer(&context, clusterDataBuffer);

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
          stagedMeshletCullBuffer.cmdCopyToDevice(
              cmd, meshlet_cull.data(),
              meshlet_cull.size() * sizeof(MeshletCullData), 0);
          stagedClusterDataBuffer.cmdCopyToDevice(
              cmd, clusterInfo.clusterInfo.data(),
              clusterInfo.clusterInfo.size() * sizeof(MeshletClusterInfo), 0);
        },
        QueueRequirement::Transfer);
  }

  // Shader
  auto msCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.mesh.spv");
  auto fsCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.frag.spv");
  auto csClearCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.clear.comp.spv");
  auto csDynLodCode =
      readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.meshlet.dynlod.comp.spv");

  ShaderModule msModule(&context, msCode, "main", ShaderStage::Mesh);
  ShaderModule fsModule(&context, fsCode, "main", ShaderStage::Fragment);
  ShaderModule csClearModule(&context, csClearCode, "main",
                             ShaderStage::Compute);
  ShaderModule csDynLodModule(&context, csDynLodCode, "main",
                              ShaderStage::Compute);

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
  auto cullBufReg = renderGraph->registerBuffer(meshletCullBuffer);
  auto swapchainResReg = renderGraph->registerImage(swapchainRes);
  auto depthReg = renderGraph->registerImage(depthImage);
  auto targetMeshletsReg = renderGraph->registerBuffer(targetMeshlets);
  auto clusterDataReg = renderGraph->registerBuffer(clusterDataBuffer);
  auto indirectDrawReg = renderGraph->registerBuffer(indirectDrawBuffer);
  auto uniformBufCullReg = renderGraph->registerBuffer(ubCullBuffer);

  auto queueReg = renderGraph->registerBuffer(productQueueBuffer);
  auto producerReg = renderGraph->registerBuffer(producerCounterBuffer);
  auto consumerReg = renderGraph->registerBuffer(consumerCounterBuffer);
  auto remainingReg = renderGraph->registerBuffer(remainingCounterBuffer);

  // Cull Pass
  auto cullPass = renderGraph->addComputePass();
  cullPass->setComputeShader(&csDynLodModule);
  cullPass->setPassDescriptorLayout(
      {DescriptorType::StorageBuffer, DescriptorType::StorageBuffer,
       DescriptorType::StorageBuffer, DescriptorType::UniformBuffer,
       DescriptorType::UniformBuffer, DescriptorType::StorageBuffer,
       DescriptorType::StorageBuffer, DescriptorType::StorageBuffer,
       DescriptorType::StorageBuffer});
  cullPass->addStorageBuffer(indirectDrawReg, 0, ResourceAccessType::Write);
  cullPass->addStorageBuffer(clusterDataReg, 1, ResourceAccessType::Read);
  cullPass->addStorageBuffer(targetMeshletsReg, 2, ResourceAccessType::Read);
  cullPass->addUniformBuffer(uniformBufCullReg, 3);
  cullPass->addUniformBuffer(uniformBufReg, 4);
  cullPass->addStorageBuffer(consumerReg, 5, ResourceAccessType::Write);
  cullPass->addStorageBuffer(producerReg, 6, ResourceAccessType::Write);
  cullPass->addStorageBuffer(queueReg, 7, ResourceAccessType::Write);
  cullPass->addStorageBuffer(remainingReg, 8, ResourceAccessType::Write);
  cullPass->setRecordFunction(
      [&](RenderPassContext *ctx) -> void { ctx->m_cmd->dispatch(1, 1, 1); });

  // Draw Pass
  // TODO: register indirect
  auto msPass = renderGraph->addGraphicsPass();
  msPass->addColorAttachment(swapchainResReg, VK_ATTACHMENT_LOAD_OP_CLEAR,
                             {{0.2f, 0.2f, 0.2f, 1.0f}});
  msPass->addStorageBuffer(msBufReg, 0, ResourceAccessType::Read);
  msPass->addStorageBuffer(msVBufReg, 1, ResourceAccessType::Read);
  msPass->addStorageBuffer(msIBufReg, 2, ResourceAccessType::Read);
  msPass->addStorageBuffer(vertexBufReg, 3, ResourceAccessType::Read);
  msPass->addStorageBuffer(cullBufReg, 4, ResourceAccessType::Read);
  msPass->addStorageBuffer(targetMeshletsReg, 5, ResourceAccessType::Read);
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
    // ctx->m_cmd->drawMeshTasks(meshlet_count, 1, 1);
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
    uniformData.mvp = mvp;
    uniformData.fov = 60 * 3.14159 / 180;
    uniformData.cameraPos = camPos;
    auto buf = ubBuffer->getActiveBuffer();
    buf->map();
    buf->copyToBuffer(&uniformData, sizeof(UniformBuffer), 0);
    buf->flush();
    buf->unmap();
    timeVal += 0.0005f;

    uniformCullData.clusterCounts = clusterInfo.clusterInfo.size();
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
int main() {
  int opt = 0;
  printf("Choose a demo to run:\n");
  printf("1. SoftRenderer\n");
  printf("2. VulkanRenderer\n");
  printf("3. Exit\n");
  while (opt != 1 && opt != 2 && opt != 3) {
    scanf("%d", &opt);
  }
  if (opt == 1) {
    return main2();
  } else if (opt == 2) {
    return demo_vulkanMeshShader();
  } else {
    return 0;
  }
}