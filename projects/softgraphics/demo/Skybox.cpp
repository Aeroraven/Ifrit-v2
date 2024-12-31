
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


#include "Skybox.h"
#include "./shader/SkyboxShaders.cuh"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "presentation/backend/AdaptiveBackendBuilder.h"
#include "presentation/window/AdaptiveWindowBuilder.h"
#include "utility/loader/ImageLoader.h"

namespace Ifrit::Demo::Skybox {
#ifdef IFRIT_FEATURE_CUDA
int mainGpu() {
  using namespace Ifrit::Display::Backend;
  using namespace Ifrit::Display::Window;
  using namespace Ifrit::SoftRenderer::TileRaster::CUDA;
  using namespace Ifrit::SoftRenderer::Utility::Loader;

  constexpr static int DEMO_RESOLUTION = 2048;

  std::shared_ptr<ImageF32> image1 =
      std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4, true);
  std::shared_ptr<ImageF32> depth =
      std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
  std::shared_ptr<TileRasterRendererCuda> renderer =
      std::make_shared<TileRasterRendererCuda>();
  FrameBuffer frameBuffer;
  VertexBuffer vertexBuffer;
  std::vector<int> indexBuffer;

  vertexBuffer.setLayout(
      {TypeDescriptors.FLOAT4, TypeDescriptors.FLOAT4, TypeDescriptors.FLOAT4});
  vertexBuffer.setVertexCount(8);
  vertexBuffer.allocateBuffer(8);

  vertexBuffer.setValue(0, 0, ifloat4(-0.1, 0.1, -0.5, 1));
  vertexBuffer.setValue(1, 0, ifloat4(-0.1, -0.1, -0.5, 1));
  vertexBuffer.setValue(2, 0, ifloat4(0.1, -0.1, -0.5, 1));
  vertexBuffer.setValue(3, 0, ifloat4(0.1, 0.1, -0.5, 1));
  vertexBuffer.setValue(4, 0, ifloat4(-0.1, 0.1, 0.5, 1));
  vertexBuffer.setValue(5, 0, ifloat4(-0.1, -0.1, 0.5, 1));
  vertexBuffer.setValue(6, 0, ifloat4(0.1, -0.1, 0.5, 1));
  vertexBuffer.setValue(7, 0, ifloat4(0.1, 0.1, 0.5, 1));

  vertexBuffer.setValue(0, 1, ifloat4(-1.0, 1.0, -1.0, 1));
  vertexBuffer.setValue(1, 1, ifloat4(-1.0, -1.0, -1.0, 1));
  vertexBuffer.setValue(2, 1, ifloat4(1.0, -1.0, -1.0, 1));
  vertexBuffer.setValue(3, 1, ifloat4(1.0, 1.0, -1.0, 1));
  vertexBuffer.setValue(4, 1, ifloat4(-1.0, 1.0, 1.0, 1));
  vertexBuffer.setValue(5, 1, ifloat4(-1.0, -1.0, 1.0, 1));
  vertexBuffer.setValue(6, 1, ifloat4(1.0, -1.0, 1.0, 1));
  vertexBuffer.setValue(7, 1, ifloat4(1.0, 1.0, 1.0, 1));

  indexBuffer = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 5, 1, 0, 0, 4, 5,
                 6, 2, 1, 1, 5, 6, 7, 3, 2, 2, 6, 7, 4, 0, 3, 3, 7, 4};

  frameBuffer.setColorAttachments({image1.get()});
  frameBuffer.setDepthAttachment(*depth);

  renderer->init();
  renderer->bindFrameBuffer(frameBuffer);
  renderer->bindVertexBuffer(vertexBuffer);
  renderer->bindIndexBuffer(indexBuffer);

  std::vector<std::vector<float>> texData(6);
  std::vector<int> texW(6), texH(6);
  ImageLoader imageLoader;
  std::array<std::string, 6> texNames = {"right.jpg",  "left.jpg",  "top.jpg",
                                         "bottom.jpg", "front.jpg", "back.jpg"};
  for (int i = 0; i < 6; i++) {
    std::string name = std::string(IFRIT_ASSET_PATH "/skybox/") + texNames[i];
    imageLoader.loadRGBA(name.c_str(), &texData[i], &texH[i], &texW[i]);
  }

  IfritImageCreateInfo imageCI;
  imageCI.extent.height = texH[0];
  imageCI.extent.width = texW[0];
  imageCI.mipLevels = 6;
  imageCI.arrayLayers = 6;
  renderer->createTexture(0, imageCI);

  for (int i = 0; i < 6; i++) {
    IfritBufferImageCopy imageCopy;
    imageCopy.imageOffset = {0, 0, 0};
    imageCopy.bufferOffset = 0;
    imageCopy.imageExtent.depth = 1;
    imageCopy.imageExtent.height = texH[i];
    imageCopy.imageExtent.width = texW[i];
    imageCopy.imageSubresource.baseArrayLayer = i;
    imageCopy.imageSubresource.mipLevel = 0;
    renderer->copyHostBufferToImage(texData[i].data(), 0, {imageCopy});
  }
  renderer->generateMipmap(0, IF_FILTER_LINEAR);

  SkyboxVS vertexShader;
  VaryingDescriptor vertexShaderLayout;
  vertexShaderLayout.setVaryingDescriptors(
      {TypeDescriptors.FLOAT4, TypeDescriptors.FLOAT4});
  SkyboxFS fragmentShader;

  IfritSamplerT sampler;
  sampler.filterMode = IF_FILTER_LINEAR;
  sampler.addressModeU = IF_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler.addressModeV = IF_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler.borderColor = IF_BORDER_COLOR_WHITE;
  sampler.anisotropyEnable = false;
  sampler.maxAnisotropy = 16.0f;
  renderer->createSampler(0, sampler);

  auto dVertexShader = vertexShader.getCudaClone();
  auto dFragmentShader = fragmentShader.getCudaClone();
  renderer->bindFragmentShader(dFragmentShader);
  renderer->bindVertexShader(dVertexShader, vertexShaderLayout);

  renderer->setDepthFunc(IF_COMPARE_OP_LESS);
  renderer->setDepthTestEnable(true);
  renderer->setClearValues({{1, 1, 1, 0}}, 255.0);

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
    renderer->clear();
    renderer->drawElements();
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    *coreTime =
        (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    backend->updateTexture(*image1);
    backend->draw();
  });
  return 0;
}
#endif

int mainCpu() { return 0; }
} // namespace Ifrit::Demo::Skybox