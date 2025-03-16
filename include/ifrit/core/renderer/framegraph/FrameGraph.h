
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

#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <functional>
#include <string>
#include <vector>

namespace Ifrit::Core {

// Migration of render graph from original RHI layer.
// Intended to making automatic layout transitions and resource management
// easier. Resource lifetime management and reuse will be considered in the
// future.
// Some references from: https://zhuanlan.zhihu.com/p/147207161

using ResourceNodeId = u32;
using PassNodeId = u32;
using FgBuffer = GraphicsBackend::Rhi::RhiBuffer;
using FgTexture = GraphicsBackend::Rhi::RhiTexture;
using FgTextureSubResource = GraphicsBackend::Rhi::RhiImageSubResource;

class FrameGraphCompiler;
class FrameGraphExecutor;

enum class FrameGraphResourceType {
  Undefined,
  ResourceBuffer,
  ResourceTexture,
};

enum class FrameGraphPassType {
  Compute,
  Graphics,
};

struct ResourceNode {
  ResourceNodeId id;
  std::string name;
  bool isImported;
  FrameGraphResourceType type;

  std::shared_ptr<FgBuffer> selfBuffer;
  std::shared_ptr<FgTexture> selfTexture;
  FgBuffer *importedBuffer;
  FgTexture *importedTexture;

  FgTextureSubResource subResource;
};

struct PassNode {
  PassNodeId id;
  FrameGraphPassType type;
  std::string name;
  bool isImported;
  std::function<void()> passFunction;
  std::vector<ResourceNodeId> inputResources;
  std::vector<ResourceNodeId> outputResources;
  std::vector<ResourceNodeId> dependentResources;
};

class IFRIT_APIDECL FrameGraph {
private:
  std::vector<ResourceNode> m_resources;
  std::vector<PassNode> m_passes;

public:
  ResourceNodeId addResource(const std::string &name);
  PassNodeId addPass(const std::string &name, FrameGraphPassType type, const std::vector<ResourceNodeId> &inputs,
                     const std::vector<ResourceNodeId> &outputs, const std::vector<ResourceNodeId> &dependencies);

  void setImportedResource(ResourceNodeId id, FgBuffer *buffer);
  void setImportedResource(ResourceNodeId id, FgTexture *texture, const FgTextureSubResource &subResource);

  void setExecutionFunction(PassNodeId id, std::function<void()> func);

  friend class FrameGraphCompiler;
  friend class FrameGraphExecutor;
};

struct CompiledFrameGraph {
  struct ResourceBarriers {
    bool enableUAVBarrier = false;
    bool enableTransitionBarrier = false;
    GraphicsBackend::Rhi::RhiResourceState2 srcState;
    GraphicsBackend::Rhi::RhiResourceState2 dstState = GraphicsBackend::Rhi::RhiResourceState2::Undefined;
  };
  const FrameGraph *m_graph = nullptr;
  std::vector<u32> m_passTopoOrder = {};
  std::vector<std::vector<ResourceBarriers>> m_passResourceBarriers = {};
  std::vector<std::vector<ResourceBarriers>> m_outputAliasedResourcesBarriers = {};
  std::vector<std::vector<u32>> m_inputResourceDependencies = {};
};

class IFRIT_APIDECL FrameGraphCompiler {
private:
public:
  CompiledFrameGraph compile(const FrameGraph &graph);
};

class IFRIT_APIDECL FrameGraphExecutor {
private:
public:
  void executeInSingleCmd(const GraphicsBackend::Rhi::RhiCommandBuffer *cmd, const CompiledFrameGraph &compiledGraph);
};

} // namespace Ifrit::Core