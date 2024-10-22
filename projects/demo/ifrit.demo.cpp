#define IFRIT_DLL
#include <common/math/LinalgOps.h>
#include <memory>

#ifndef _MSC_VER
#include <softrenderer/include/engine/tileraster/TileRasterRenderer.h>
#endif

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

#ifndef _MSC_VER
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
  std::unique_ptr<ImageF32> color = std::make_unique<ImageF32>(800, 600, 4);
  std::unique_ptr<ImageF32> depth = std::make_unique<ImageF32>(800, 600, 1);
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

  using namespace Ifrit::Presentation::Backend;
  using namespace Ifrit::Presentation::Window;
  GLFWWindowProvider windowProvider;
  windowProvider.setup(800, 600);
  windowProvider.setTitle("Ifrit");

  OpenGLBackend backend;
  backend.setViewport(0, 0, 800, 600);

  windowProvider.loop([&](int *repCore) {
    renderer->drawElements(3, true);
    backend.updateTexture(color->getData(), 4, 800, 600);
    backend.draw();
  });
  return 0;
}
#endif

int main3() {
  using namespace Ifrit::Engine::VkRenderer;
  using namespace Ifrit::Presentation::Window;
  using namespace Ifrit::Math;

  GLFWWindowProviderInitArgs glfwArgs;
  glfwArgs.vulkanMode = true;
  GLFWWindowProvider windowProvider(glfwArgs);
  windowProvider.setup(800, 600);
  windowProvider.setTitle("Ifrit");

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

  EngineContext context(args);
  Swapchain swapchain(&context);

  // Print swapchain queue
  auto vsCode = readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.demo.vert.spv");
  auto fsCode = readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.demo.frag.spv");

  ShaderModule vsModule(&context, vsCode, "main", ShaderStage::Vertex);
  ShaderModule fsModule(&context, fsCode, "main", ShaderStage::Fragment);

  Viewport viewport = {0.0f, 0.0f, 800.0f, 600.0f, 0.0f, 1.0f};
  Scissor scissor = {0, 0, 800, 600};

  std::vector<float> vertexData = {-0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.5f, -0.5f,
                                   0.0f,  1.0f,  0.0f, 0.5f, 0.5f, 0.0f, 0.0f,
                                   1.0f,  -0.5f, 0.5f, 1.0f, 1.0f, 1.0f};
  std::vector<uint32_t> indexData = {0, 1, 2, 2, 3, 0};

  // Resource manager
  DescriptorManager bindlessDesc(&context);
  ResourceManager resourceManager(&context);
  CommandExecutor backend(&context, &swapchain, &bindlessDesc,
                          &resourceManager);
  backend.setQueues(true, 1, 0, 1);

  // Vertex buffer & index buffer
  VertexBufferDescriptor vbDesc;
  vbDesc.addBinding({0, 1},
                    {VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT},
                    {0, 8}, 20, VertexInputRate::Vertex);

  BufferCreateInfo vbCI{};
  vbCI.size = vertexData.size() * sizeof(float);
  vbCI.usage =
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  BufferCreateInfo ibCI{};
  ibCI.size = indexData.size() * sizeof(uint32_t);
  ibCI.usage =
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  auto vxBuffer = resourceManager.createSimpleBuffer(vbCI);
  auto ixBuffer = resourceManager.createSimpleBuffer(ibCI);

  StagedSingleBuffer stagedVertexBuffer(&context, vxBuffer);
  StagedSingleBuffer stagedIndexBuffer(&context, ixBuffer);
  backend.runImmidiateCommand(
      [&stagedVertexBuffer, &vertexData](CommandBuffer *cmd) {
        stagedVertexBuffer.cmdCopyToDevice(
            cmd, vertexData.data(), vertexData.size() * sizeof(float), 0);
      },
      QueueRequirement::Transfer);
  backend.runImmidiateCommand(
      [&stagedIndexBuffer, &indexData](CommandBuffer *cmd) {
        stagedIndexBuffer.cmdCopyToDevice(
            cmd, indexData.data(), indexData.size() * sizeof(uint32_t), 0);
      },
      QueueRequirement::Transfer);

  // Uniform Buffer
  BufferCreateInfo ubCI{};
  ubCI.size = 16 * sizeof(float);
  ubCI.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  ubCI.hostVisible = true;

  auto ubBuffer = resourceManager.createTracedMultipleBuffer(ubCI);

  // Render graph
  auto renderGraph = backend.createRenderGraph();
  auto pass = renderGraph->addGraphicsPass();
  auto swapchainRes = backend.getSwapchainImageResource();
  auto swapchainResReg = renderGraph->registerImage(swapchainRes);
  auto vbReg = renderGraph->registerBuffer(stagedVertexBuffer.getBuffer());
  auto ibReg = renderGraph->registerBuffer(stagedIndexBuffer.getBuffer());
  auto ubReg = renderGraph->registerBuffer(ubBuffer);

  pass->addColorAttachment(swapchainResReg, VK_ATTACHMENT_LOAD_OP_CLEAR,
                           {{0.2f, 0.2f, 0.2f, 1.0f}});
  pass->setVertexShader(&vsModule);
  pass->setFragmentShader(&fsModule);
  pass->setRenderArea(0, 0, 800, 600);
  pass->setVertexInput(vbDesc, {vbReg});
  pass->setIndexInput(ibReg, VK_INDEX_TYPE_UINT32);

  pass->addUniformBuffer(ubReg, 0);
  pass->setPassDescriptorLayout({DescriptorType::UniformBuffer});

  // Uniform data
  float4x4 view = (lookAt({0, 0.1, 2.25}, {0, 0.0, 0.0}, {0, 1, 0}));
  float4x4 proj = (perspective(60 * 3.14159 / 180, 800.0 / 600.0, 0.01, 3000));

  // Pass Record
  pass->setRecordFunction([&](RenderPassContext *ctx) -> void {
    ctx->m_cmd->setScissors({scissor});
    ctx->m_cmd->setViewports({viewport});
    ctx->m_cmd->drawIndexed(6, 1, 0, 0, 0);
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
    swapchain.acquireNextImage();
    backend.runRenderGraph(renderGraph);
    swapchain.presentImage();
    auto endTime = std::chrono::high_resolution_clock::now();
    *repCore = std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                                     startTime)
                   .count();
  });
  context.waitIdle();
  return 0;
}

int main() {
#ifdef _MSC_VER
  return main3();
#else
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
    return main3();
  } else {
    return 0;
  }
#endif
}