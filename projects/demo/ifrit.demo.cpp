#define IFRIT_DLL
#include <memory>
#include <softrenderer/include/engine/tileraster/TileRasterRenderer.h>

#include <display/include/presentation/backend/OpenGLBackend.h>
#include <display/include/presentation/window/GLFWWindowProvider.h>

#include <vkrenderer/include/engine/vkrenderer/Command.h>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vkrenderer/include/engine/vkrenderer/RenderGraph.h>
#include <vkrenderer/include/engine/vkrenderer/Shader.h>
#include <vkrenderer/include/engine/vkrenderer/Swapchain.h>

#include <chrono>

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

int main3() {
  using namespace Ifrit::Engine::VkRenderer;
  using namespace Ifrit::Presentation::Window;
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
  auto presentQueue = swapchain.getPresentQueue();
  printf("Present queue: %p\n", presentQueue);

  QueueCollections queueCollections(&context);
  queueCollections.loadQueues();
  auto graphicsQueues = queueCollections.getGraphicsQueues();

  std::vector<Queue *> queues;
  queues.push_back(graphicsQueues[0]);
  if (presentQueue != graphicsQueues[0]->getQueue()) {
    bool found = false;
    for (int i = 0; i < graphicsQueues.size(); i++) {
      if (graphicsQueues[i]->getQueue() == presentQueue) {
        queues.push_back(graphicsQueues[i]);
        found = true;
        break;
      }
    }
    if (!found) {
      printf("Present queue not found in graphics queues\n");
      return 1;
    }
  }

  auto vsCode = readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.demo.vert.spv");
  auto fsCode = readShaderFile(IFRIT_DEMO_SHADER_PATH "/ifrit.demo.frag.spv");

  ShaderModuleCI vsCI;
  vsCI.code = vsCode;
  vsCI.stage = ShaderStage::Vertex;
  vsCI.entryPoint = "main";

  ShaderModuleCI fsCI;
  fsCI.code = fsCode;
  fsCI.stage = ShaderStage::Fragment;
  fsCI.entryPoint = "main";

  ShaderModule vsModule(&context, vsCI);
  ShaderModule fsModule(&context, fsCI);

  Viewport viewport;
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = 800.0f;
  viewport.height = 600.0f;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  Scissor scissor;
  scissor.x = 0;
  scissor.y = 0;
  scissor.width = 800;
  scissor.height = 600;


  RenderGraphExecutor backend(&context, &swapchain);
  RenderGraph renderGraph(&context);
  auto swapchainRes = backend.getSwapchainImageResource();
  auto pass = renderGraph.addGraphicsPass();
  auto swapchainResReg = renderGraph.registerSwapchainImage(swapchainRes);
  pass->addColorAttachment(swapchainResReg, VK_ATTACHMENT_LOAD_OP_CLEAR,
                           {{0.0f, 1.0f, 0.0f, 1.0f}});
  pass->setVertexShader(&vsModule);
  pass->setFragmentShader(&fsModule);
  pass->setRenderArea(0, 0, 800, 600);
  pass->setRecordFunction([scissor,viewport](RenderPassContext *ctx) -> void {
    ctx->m_cmd->setScissors({scissor});
    ctx->m_cmd->setViewports({viewport});
    ctx->m_cmd->draw(3, 1, 0, 0);
  });
  renderGraph.build();
  backend.setQueues(queues);
  backend.compileGraph(&renderGraph);


  windowProvider.loop([&](int *repCore) {
    auto startTime = std::chrono::high_resolution_clock::now();
    swapchain.acquireNextImage();
    backend.runGraph(&renderGraph);
    swapchain.presentImage();
    auto endTime = std::chrono::high_resolution_clock::now();
    *repCore = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
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
    return main3();
  } else {
    return 0;
  }
}