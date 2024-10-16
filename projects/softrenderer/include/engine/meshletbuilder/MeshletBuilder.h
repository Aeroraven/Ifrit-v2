#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/base/VertexBuffer.h"
#include "MeshletCommon.h"

namespace Ifrit::Engine::SoftRenderer::MeshletBuilder {
class IFRIT_APIDECL TrivialMeshletBuilder {
private:
  const VertexBuffer *vbuffer = nullptr;
  const std::vector<int> *ibuffer = nullptr;

public:
  void bindVertexBuffer(const VertexBuffer &vbuffer);
  void bindIndexBuffer(const std::vector<int> &ibuffer);
  void buildMeshlet(int posAttrId,
                    std::vector<std::unique_ptr<Meshlet>> &outData);
  void mergeMeshlet(const std::vector<std::unique_ptr<Meshlet>> &meshlets,
                    Meshlet &outData, std::vector<int> &outVertexOffset,
                    std::vector<int> &outIndexOffset, bool autoIncre);
};
} // namespace Ifrit::Engine::SoftRenderer::MeshletBuilder