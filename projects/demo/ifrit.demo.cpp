#ifndef IFRIT_DLL
#define IFRIT_DLL
#endif
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/Core.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"

#define WINDOW_WIDTH 1980
#define WINDOW_HEIGHT 1080

using namespace Ifrit::GraphicsBackend::Rhi;
using namespace Ifrit::MeshProcLib::ClusterLod;
using namespace Ifrit::Math;
using namespace Ifrit::Core;
using namespace Ifrit::Common::Utility;

// Glfw key function here
float movLeft = 0, movRight = 0, movTop = 0, movBottom = 0, movFar = 0,
      movNear = 0;

void key_callback(int key, int scancode, int action, int mods) {
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

class DemoApplication : public Ifrit::Core::Application {
private:
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

  RhiGraphicsPass *msPass;
  RhiComputePass *cullPass;
  RhiMultiBuffer *ubBuffer;
  RhiMultiBuffer *ubCullBuffer;
  RhiBuffer *indirectDrawBuffer;
  RhiTexture *swapchainImg;

  constexpr static int MAX_LOD = 4;

  RhiBindlessDescriptorRef *msDescriptor;
  RhiBindlessDescriptorRef *csDescriptor;

  std::shared_ptr<RhiRenderTargets> renderTargets;
  std::shared_ptr<RhiColorAttachment> colorAttachment;
  std::shared_ptr<RhiDepthStencilAttachment> depthAttachment;

  std::shared_ptr<MeshData> meshData;

public:
  void onStart() override {
    auto obj = m_assetManager->getAssetByName<WaveFrontAsset>("bunny.obj");
    if (m_sceneAssetManager->checkSceneExists("TestScene2")) {
      auto t = m_sceneAssetManager->getScene("TestScene2");
    } else {
      auto s = m_sceneAssetManager->createScene("TestScene2");
      auto node = s->addSceneNode();
      auto gameObject = node->addGameObject();
      auto meshFilter = gameObject->addComponent<MeshFilter>();

      meshFilter->setMesh(obj);
      m_sceneAssetManager->saveScenes();
    }

    m_windowProvider->registerKeyCallback(key_callback);
    auto rt = m_rhiLayer.get();
    // load meshlet
    std::vector<ifloat3> vertices;
    std::vector<ifloat4> verticesAligned;
    std::vector<uint32_t> indices;

    meshData = obj->loadMesh();
    obj->createMeshLodHierarchy(meshData);
    indices = meshData->m_indices;
    vertices = meshData->m_vertices;
    verticesAligned = meshData->m_verticesAligned;

    const size_t max_vertices = 64;
    const size_t max_triangles = 124;
    const float cone_weight = 0.0f;

    auto meshletBuffer = rt->createStorageBufferDevice(
        size_cast<uint32_t>(meshData->m_meshlets.size() * sizeof(iint4)),
        RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto meshletVertexBuffer = rt->createStorageBufferDevice(
        size_cast<int>(meshData->m_meshletVertices.size() *
                       sizeof(unsigned int)),
        RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto meshletIndexBuffer = rt->createStorageBufferDevice(
        size_cast<int>(meshData->m_meshletTriangles.size() *
                       sizeof(unsigned int)),
        RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto vertexBuffer = rt->createStorageBufferDevice(
        size_cast<int>(meshData->m_verticesAligned.size() * sizeof(ifloat4)),
        RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    ubBuffer = rt->createUniformBufferShared(
        size_cast<int>(sizeof(UniformBuffer)), true, 0);

    // Culling pipeline
    indirectDrawBuffer = rt->createIndirectMeshDrawBufferDevice(1);
    auto filteredMeshletBuffer = rt->createStorageBufferDevice(
        size_cast<int>(meshData->m_meshlets.size() * sizeof(uint32_t)), 0);
    auto bvhNodeBuffer = rt->createStorageBufferDevice(
        size_cast<int>(meshData->m_bvhNodes.size() * sizeof(FlattenedBVHNode)),
        RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto clusterGroupBuffer = rt->createStorageBufferDevice(
        size_cast<int>(meshData->m_clusterGroups.size() * sizeof(ClusterGroup)),
        RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto consumerCounterBuffer =
        rt->createStorageBufferDevice(size_cast<int>(sizeof(uint32_t)), 0);
    auto producerCounterBuffer =
        rt->createStorageBufferDevice(size_cast<int>(sizeof(uint32_t)), 0);
    auto remainingCounterBuffer =
        rt->createStorageBufferDevice(size_cast<int>(sizeof(uint32_t)), 0);
    auto productQueueBuffer = rt->createStorageBufferDevice(
        size_cast<int>(sizeof(uint32_t) * meshData->m_bvhNodes.size()), 0);
    ubCullBuffer = rt->createUniformBufferShared(
        size_cast<int>(sizeof(UniformCullBuffer)), true, 0);
    auto meshletInClusterBuffer = rt->createStorageBufferDevice(
        size_cast<int>(meshData->m_meshletInClusterGroup.size() *
                       sizeof(uint32_t)),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    uniformData.meshletCount = size_cast<uint32_t>(meshData->m_meshlets.size());

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
      auto stagedMeshletInClusterBuffer =
          rt->createStagedSingleBuffer(meshletInClusterBuffer);

      auto tq = rt->getQueue(RHI_QUEUE_TRANSFER_BIT);

      tq->runSyncCommand([&](const RhiCommandBuffer *cmd) -> void {
        stagedMeshletBuffer->cmdCopyToDevice(
            cmd, meshData->m_meshlets.data(),
            size_cast<uint32_t>(meshData->m_meshlets.size() * sizeof(iint4)),
            0);
        stagedMeshletVertexBuffer->cmdCopyToDevice(
            cmd, meshData->m_meshletVertices.data(),
            size_cast<uint32_t>(meshData->m_meshletVertices.size() *
                                sizeof(unsigned int)),
            0);
        stagedMeshletIndexBuffer->cmdCopyToDevice(
            cmd, meshData->m_meshletTriangles.data(),
            size_cast<uint32_t>(meshData->m_meshletTriangles.size() *
                                sizeof(unsigned int)),
            0);
        stagedVertexBuffer->cmdCopyToDevice(
            cmd, verticesAligned.data(),
            size_cast<uint32_t>(verticesAligned.size() * sizeof(ifloat4)), 0);
        stagedBvhNodeBuffer->cmdCopyToDevice(
            cmd, meshData->m_bvhNodes.data(),
            size_cast<uint32_t>(meshData->m_bvhNodes.size() *
                                sizeof(FlattenedBVHNode)),
            0);
        stagedClusterGroupBuffer->cmdCopyToDevice(
            cmd, meshData->m_clusterGroups.data(),
            size_cast<uint32_t>(meshData->m_clusterGroups.size() *
                                sizeof(ClusterGroup)),
            0);
        stagedMeshletInClusterBuffer->cmdCopyToDevice(
            cmd, meshData->m_meshletInClusterGroup.data(),
            size_cast<uint32_t>(meshData->m_meshletInClusterGroup.size() *
                                sizeof(uint32_t)),
            0);
      });
    }

    // Shader
    auto msModule =
        m_assetManager
            ->getAssetByName<ShaderAsset>("Shader/ifrit.meshlet.mesh.glsl")
            ->loadShader();
    auto fsModule =
        m_assetManager
            ->getAssetByName<ShaderAsset>("Shader/ifrit.meshlet.frag.glsl")
            ->loadShader();
    auto csDynLodModule = m_assetManager
                              ->getAssetByName<ShaderAsset>(
                                  "Shader/ifrit.meshlet.dynlod.comp.glsl")
                              ->loadShader();

    // Render targets
    auto depthImage = rt->createDepthRenderTexture(WINDOW_WIDTH, WINDOW_HEIGHT);
    swapchainImg = rt->getSwapchainImage();
    renderTargets = rt->createRenderTargets();
    colorAttachment = rt->createRenderTarget(
        swapchainImg, {0.0f, 0.0f, 0.0f, 1.0f}, RhiRenderTargetLoadOp::Clear);
    depthAttachment = rt->createRenderTargetDepthStencil(
        depthImage, {{}, 1.0f}, RhiRenderTargetLoadOp::Clear);
    renderTargets->setColorAttachments({colorAttachment.get()});
    renderTargets->setDepthStencilAttachment(depthAttachment.get());
    renderTargets->setRenderArea(scissor);

    // Cull Pass
    cullPass = rt->createComputePass();
    csDescriptor = rt->createBindlessDescriptorRef();
    csDescriptor->addStorageBuffer(indirectDrawBuffer, 0);
    csDescriptor->addStorageBuffer(clusterGroupBuffer, 1);
    csDescriptor->addStorageBuffer(bvhNodeBuffer, 2);
    csDescriptor->addStorageBuffer(filteredMeshletBuffer, 3);
    csDescriptor->addStorageBuffer(meshletInClusterBuffer, 4);
    csDescriptor->addUniformBuffer(ubCullBuffer, 5);
    csDescriptor->addUniformBuffer(ubBuffer, 6);
    csDescriptor->addStorageBuffer(consumerCounterBuffer, 7);
    csDescriptor->addStorageBuffer(producerCounterBuffer, 8);
    csDescriptor->addStorageBuffer(productQueueBuffer, 9);
    csDescriptor->addStorageBuffer(remainingCounterBuffer, 10);

    cullPass->setComputeShader(csDynLodModule);
    cullPass->setNumBindlessDescriptorSets(1);
    cullPass->setRecordFunction([&](RhiRenderPassContext *ctx) -> void {
      ctx->m_cmd->attachBindlessReferenceCompute(cullPass, 1, csDescriptor);
      ctx->m_cmd->dispatch(1, 1, 1);
    });

    // Draw Pass
    // TODO: register indirect
    msPass = rt->createGraphicsPass();

    msDescriptor = rt->createBindlessDescriptorRef();
    msDescriptor->addStorageBuffer(meshletBuffer, 0);
    msDescriptor->addStorageBuffer(meshletVertexBuffer, 1);
    msDescriptor->addStorageBuffer(meshletIndexBuffer, 2);
    msDescriptor->addStorageBuffer(vertexBuffer, 3);
    msDescriptor->addStorageBuffer(filteredMeshletBuffer, 4);
    msDescriptor->addUniformBuffer(ubBuffer, 5);

    msPass->setNumBindlessDescriptorSets(1);
    msPass->setMeshShader(msModule);
    msPass->setPixelShader(fsModule);
    msPass->setRasterizerTopology(RhiRasterizerTopology::TriangleList);
    msPass->setDepthAttachment(depthImage, RhiRenderTargetLoadOp::Clear,
                               {{}, 1.0f});

    msPass->setDepthCompareOp(RhiCompareOp::Less);
    msPass->setDepthWrite(true);
    msPass->setDepthTestEnable(true);

    msPass->setRecordFunction([&](RhiRenderPassContext *ctx) -> void {
      ctx->m_cmd->attachBindlessReferenceGraphics(msPass, 1, msDescriptor);
      ctx->m_cmd->drawMeshTasksIndirect(indirectDrawBuffer, 0, 1, 0);
    });
    msPass->setRecordFunctionPostRenderPass(
        [&](RhiRenderPassContext *ctx) -> void {
          ctx->m_cmd->imageBarrier(swapchainImg, RhiResourceState::RenderTarget,
                                   RhiResourceState::Present);
        });

    msPass->setExecutionFunction([&](RhiRenderPassContext *ctx) -> void {
      float z = 0.25f + (movFar - movNear) * 0.02f; // + 0.5*sin(timeVal);
      float x = 0 + (movRight - movLeft) * 0.02f;
      float y = 0.1f + (movTop - movBottom) * 0.02f;

      ifloat4 camPos = ifloat4(x, y, z, 1.0f);
      float4x4 view = (lookAt({camPos.x, camPos.y, z},
                              {camPos.x, camPos.y, z - 1}, {0, 1, 0}));
      float4x4 proj = (perspectiveNegateY(60 * 3.14159f / 180,
                                          1.0f * WINDOW_WIDTH / WINDOW_HEIGHT,
                                          0.01f, 3000.0f));
      auto mvp = transpose(matmul(proj, view));
      auto mv = transpose(view);
      uniformData.mvp = mvp;
      uniformData.mv = mv;
      uniformData.fov = 60 * 3.14159f / 180;
      uniformData.cameraPos = camPos;
      auto buf = ubBuffer->getActiveBuffer();
      buf->map();
      buf->writeBuffer(&uniformData, sizeof(UniformBuffer), 0);
      buf->flush();
      buf->unmap();

      uniformCullData.totalBvhNodes =
          size_cast<uint32_t>(meshData->m_bvhNodes.size());
      uniformCullData.clusterGroupCounts =
          size_cast<uint32_t>(meshData->m_clusterGroups.size());
      uniformCullData.totalLods = MAX_LOD;
      auto buf2 = ubCullBuffer->getActiveBuffer();
      buf2->map();
      buf2->writeBuffer(&uniformCullData, sizeof(UniformCullBuffer), 0);
      buf2->flush();
      buf2->unmap();
    });
  }

  void onUpdate() override {
    auto rt = m_rhiLayer.get();
    auto compQueue = rt->getQueue(RHI_QUEUE_COMPUTE_BIT);
    auto drawQueue = rt->getQueue(RHI_QUEUE_GRAPHICS_BIT);
    rt->beginFrame();
    auto sFrameReady = rt->getSwapchainFrameReadyEventHandler();
    auto sRenderComplete = rt->getSwapchainRenderDoneEventHandler();
    auto sCompEnd = compQueue->runAsyncCommand(
        [&](const RhiCommandBuffer *cmd) { cullPass->run(cmd, 0); },
        {sFrameReady.get()}, {});
    auto sDrawEnd = drawQueue->runAsyncCommand(
        [&](const RhiCommandBuffer *cmd) {
          msPass->run(cmd, renderTargets.get(), 0);
        },
        {sCompEnd.get()}, {sRenderComplete.get()});
    rt->endFrame();
  }

  void onEnd() override {}
};

int main() {
  Ifrit::Core::ApplicationCreateInfo info;
  info.m_assetPath = IFRIT_DEMO_ASSET_PATH;
  info.m_scenePath = IFRIT_DEMO_SCENE_PATH;
  info.m_displayProvider = Ifrit::Core::ApplicationDisplayProvider::GLFW;
  info.m_rhiType = Ifrit::Core::ApplicationRhiType::Vulkan;
  info.m_width = 1980;
  info.m_height = 1080;
  info.m_rhiComputeQueueCount = 1;
  info.m_rhiGraphicsQueueCount = 1;
  info.m_rhiTransferQueueCount = 1;
  info.m_rhiNumBackBuffers = 2;
  info.m_name = "Ifrit-v2";

  DemoApplication app;
  app.run(info);
  return 0;
}