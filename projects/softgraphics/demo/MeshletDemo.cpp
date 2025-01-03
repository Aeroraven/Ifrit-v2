
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

#include "MeshletDemo.h"
#include "./demo/shader/DefaultDemoShaders.cuh"
#include "core/data/Image.h"
#include "engine/bufferman/BufferManager.h"
#include "engine/meshletbuilder/MeshletBuilder.h"
#include "engine/tileraster/TileRasterRenderer.h"
#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "ifrit/common/math/LinalgOps.h"
#include "presentation/backend/AdaptiveBackendBuilder.h"
#include "presentation/backend/TerminalAsciiBackend.h"
#include "presentation/backend/TerminalCharColorBackend.h"
#include "presentation/window/AdaptiveWindowBuilder.h"
#include "shader/MeshletDemoShaders.cuh"
#include "utility/loader/ImageLoader.h"
#include "utility/loader/WavefrontLoader.h"

#define DEMO_RESOLUTION 1800
namespace Ifrit::Demo::MeshletDemo {
using namespace std;
using namespace Ifrit::SoftRenderer::Core::Data;
using namespace Ifrit::SoftRenderer::TileRaster;
using namespace Ifrit::SoftRenderer::Utility::Loader;
using namespace Ifrit::Math;
using namespace Ifrit::Display::Window;
using namespace Ifrit::Display::Backend;
using namespace Ifrit::SoftRenderer::MeshletBuilder;
using namespace Ifrit::SoftRenderer::BufferManager;
#ifdef IFRIT_FEATURE_CUDA
using namespace Ifrit::SoftRenderer::TileRaster::CUDA;
using namespace Ifrit::SoftRenderer::Core::CUDA;
using namespace Ifrit::SoftRenderer::TileRaster::CUDA::Invocation;
#endif

float4x4 view = (lookAt({0, 0.1, 0.25}, {0, 0.1, 0.0}, {0, 1, 0}));
// float4x4 view = (lookAt({ 0,0.75,1.50 }, { 0,0.75,0.0 }, { 0,1,0 }));
float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.01, 3000));
float4x4 mvp = matmul(proj, view);

class MeshletDemoVS : public VertexShader {
public:
  IFRIT_DUAL virtual void execute(const void *const *input, ifloat4 *outPos,
                                  ifloat4 *const *outVaryings) override {
    auto s = *reinterpret_cast<const ifloat4 *>(input[0]);
    auto p = matmul(mvp, s);
    *outPos = p;
    *outVaryings[0] = *reinterpret_cast<const ifloat4 *>(input[1]);
  }
};

class MeshletDemoFS : public FragmentShader {
public:
  IFRIT_DUAL virtual void execute(const void *varyings, void *colorOutput,
                                  float *fragmentDepth) override {
    ifloat4 result = ((const VaryingStore *)varyings)[0].vf4;
    constexpr float fw = 0.5;
    constexpr float ds = 1.0;
    result.x = fw * result.x * ds + fw * ds;
    result.y = fw * result.y * ds + fw * ds;
    result.z = fw * result.z * ds + fw * ds;
    result.w = 0.5;
    auto &co = ((ifloat4 *)colorOutput)[0];
    co = result;
  }
};

#ifdef IFRIT_FEATURE_CUDA
int mainGpu() {
  WavefrontLoader loader;
  std::vector<ifloat3> pos;
  std::vector<ifloat3> normal;
  std::vector<ifloat2> uv;
  std::vector<uint32_t> index;
  std::vector<ifloat3> procNormal;

  loader.loadObject(IFRIT_ASSET_PATH "/bunny.obj", pos, normal, uv, index);
  procNormal = loader.remapNormals(normal, index, pos.size());

  std::shared_ptr<ImageF32> image =
      std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4);
  std::shared_ptr<ImageF32> depth =
      std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
  std::shared_ptr<TileRasterRendererCuda> renderer =
      std::make_shared<TileRasterRendererCuda>();
  FrameBuffer frameBuffer;

  VertexBuffer vertexBuffer;
  vertexBuffer.setLayout({TypeDescriptors.FLOAT4, TypeDescriptors.FLOAT4});
  vertexBuffer.allocateBuffer(pos.size());
  for (int i = 0; i < pos.size(); i++) {
    vertexBuffer.setValue(i, 0, ifloat4(pos[i].x, pos[i].y, pos[i].z, 1));
    vertexBuffer.setValue(
        i, 1, ifloat4(procNormal[i].x, procNormal[i].y, procNormal[i].z, 0));
  }
  std::vector<int> indexBuffer = {0, 1, 2, 2, 3, 0};
  indexBuffer.resize(index.size() / 3);
  for (int i = 0; i < index.size(); i += 3) {
    indexBuffer[i / 3] = index[i];
  }

  TrivialMeshletBuilder mBuilder;
  mBuilder.bindIndexBuffer(indexBuffer);
  mBuilder.bindVertexBuffer(vertexBuffer);

  std::vector<std::unique_ptr<Meshlet>> outMeshlet;

  std::vector<int> outVertOffset, outIndexOffset;

  Meshlet mergedMeshlet;
  mBuilder.buildMeshlet(0, outMeshlet);
  mBuilder.mergeMeshlet(outMeshlet, mergedMeshlet, outVertOffset,
                        outIndexOffset, false);
  int totalMeshlets = outMeshlet.size(), totalInds = mergedMeshlet.ibufs.size(),
      totalVerts = mergedMeshlet.vbufs.getVertexCount();
  frameBuffer.setColorAttachments({image.get()});
  frameBuffer.setDepthAttachment(*depth);

  renderer->init();
  renderer->bindFrameBuffer(frameBuffer);

  renderer->createBuffer(0, mergedMeshlet.vbufs.getVertexCount() * 2 *
                                sizeof(ifloat4));
  renderer->createBuffer(1, mergedMeshlet.ibufs.size() * sizeof(int));
  renderer->createBuffer(2, outVertOffset.size() * sizeof(int));
  renderer->createBuffer(3, outIndexOffset.size() * sizeof(int));

  renderer->copyHostBufferToBuffer(
      mergedMeshlet.vbufs.getValuePtr<char>(0, 0), 0,
      mergedMeshlet.vbufs.getVertexCount() * 2 * sizeof(ifloat4));
  renderer->copyHostBufferToBuffer(mergedMeshlet.ibufs.data(), 1,
                                   mergedMeshlet.ibufs.size() * sizeof(int));
  renderer->copyHostBufferToBuffer(outVertOffset.data(), 2,
                                   outVertOffset.size() * sizeof(int));
  renderer->copyHostBufferToBuffer(outIndexOffset.data(), 3,
                                   outIndexOffset.size() * sizeof(int));

  VaryingDescriptor vertexShaderLayout;
  vertexShaderLayout.setVaryingDescriptors({TypeDescriptors.FLOAT4});

  MeshletDemoCuMS meshShader;
  MeshletDemoCuTS taskShader;
  MeshletDemoCuFS fragmentShader;
  fragmentShader.allowDepthModification = false;
  auto dMeshShader = meshShader.getCudaClone();
  auto dTaskShader = taskShader.getCudaClone();
  auto dFragmentShader = fragmentShader.getCudaClone();

  renderer->bindFragmentShader(dFragmentShader);
  renderer->bindMeshShader(dMeshShader, vertexShaderLayout, {1, 1, 1});
  renderer->bindTaskShader(dTaskShader, vertexShaderLayout);
  renderer->setClearValues({{1, 1, 1, 1}}, 255.0);

  renderer->setScissors({{0, 0, DEMO_RESOLUTION / 2, DEMO_RESOLUTION / 2},
                         {DEMO_RESOLUTION / 2, DEMO_RESOLUTION / 2,
                          DEMO_RESOLUTION, DEMO_RESOLUTION}});
  renderer->setScissorTestEnable(false);

  auto windowBuilder = std::make_unique<AdaptiveWindowBuilder>();
  auto windowProvider = windowBuilder->buildUniqueWindowProvider();
  windowProvider->setup(2048, 1152);
  windowProvider->setTitle("Ifrit-v2 <Mesh Shader>");

  auto backendBuilder = std::make_unique<AdaptiveBackendBuilder>();
  auto backend = backendBuilder->buildUniqueBackend();

  backend->setViewport(0, 0, windowProvider->getWidth(),
                       windowProvider->getHeight());
  windowProvider->loop([&](int *coreTime) {
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    renderer->clear();
    renderer->drawMeshTasks(totalMeshlets, 0);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    *coreTime =
        (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    backend->updateTexture(*image);
    backend->draw();
  });

  printf("Done\n");
  return 0;
}
#endif
int mainCpu() {
  WavefrontLoader loader;
  std::vector<ifloat3> pos;
  std::vector<ifloat3> normal;
  std::vector<ifloat2> uv;
  std::vector<uint32_t> index;
  std::vector<ifloat3> procNormal;

  loader.loadObject(IFRIT_ASSET_PATH "/bunny.obj", pos, normal, uv, index);
  procNormal = loader.remapNormals(normal, index, pos.size());

  std::shared_ptr<ImageF32> image =
      std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4);
  std::shared_ptr<ImageF32> depth =
      std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
  std::shared_ptr<TileRasterRenderer> renderer =
      std::make_shared<TileRasterRenderer>();
  FrameBuffer frameBuffer;

  VertexBuffer vertexBuffer;
  vertexBuffer.setLayout({TypeDescriptors.FLOAT4, TypeDescriptors.FLOAT4});
  vertexBuffer.allocateBuffer(pos.size());
  for (int i = 0; i < pos.size(); i++) {
    vertexBuffer.setValue(i, 0, ifloat4(pos[i].x, pos[i].y, pos[i].z, 1));
    vertexBuffer.setValue(
        i, 1, ifloat4(procNormal[i].x, procNormal[i].y, procNormal[i].z, 0));
  }
  std::vector<int> indexBuffer = {0, 1, 2, 2, 3, 0};
  indexBuffer.resize(index.size() / 3);
  for (int i = 0; i < index.size(); i += 3) {
    indexBuffer[i / 3] = index[i];
  }

  TrivialMeshletBuilder mBuilder;
  mBuilder.bindIndexBuffer(indexBuffer);
  mBuilder.bindVertexBuffer(vertexBuffer);

  std::vector<std::unique_ptr<Meshlet>> outMeshlet;
  std::vector<int> outVertOffset, outIndexOffset;
  Meshlet mergedMeshlet;
  printf("Prepare to build\n");
  mBuilder.buildMeshlet(0, outMeshlet);
  printf("Built\n");
  mBuilder.mergeMeshlet(outMeshlet, mergedMeshlet, outVertOffset,
                        outIndexOffset, true);

  frameBuffer.setColorAttachments({image.get()});
  frameBuffer.setDepthAttachment(*depth);

  renderer->init();
  renderer->bindFrameBuffer(frameBuffer);
  renderer->bindVertexBuffer(mergedMeshlet.vbufs);

  shared_ptr<TrivialBufferManager> bufferman =
      make_shared<TrivialBufferManager>();
  bufferman->init();
  auto indexBuffer1 = bufferman->createBuffer(
      {sizeof(mergedMeshlet.ibufs[0]) * mergedMeshlet.ibufs.size()});
  bufferman->bufferData(indexBuffer1, indexBuffer.data(), 0,
                        sizeof(mergedMeshlet.ibufs[0]) *
                            mergedMeshlet.ibufs.size());

  renderer->bindIndexBuffer(indexBuffer1);
  renderer->optsetForceDeterministic(true);

  MeshletDemoVS vertexShader;
  VaryingDescriptor vertexShaderLayout;
  vertexShaderLayout.setVaryingDescriptors({TypeDescriptors.FLOAT4});
  renderer->bindVertexShaderLegacy(vertexShader, vertexShaderLayout);
  MeshletDemoFS fragmentShader;
  renderer->bindFragmentShader(fragmentShader);

  auto windowBuilder = std::make_unique<AdaptiveWindowBuilder>();
  auto windowProvider = windowBuilder->buildUniqueWindowProvider();
  windowProvider->setup(2048, 1152);

  auto backendBuilder = std::make_unique<AdaptiveBackendBuilder>();
  auto backend = backendBuilder->buildUniqueBackend();

  backend->setViewport(0, 0, windowProvider->getWidth(),
                       windowProvider->getHeight());
  windowProvider->loop([&](int *coreTime) {
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    renderer->drawElements(mergedMeshlet.ibufs.size(), true);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    *coreTime =
        (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    backend->updateTexture(*image);
    backend->draw();
  });
  return 0;
}
} // namespace Ifrit::Demo::MeshletDemo