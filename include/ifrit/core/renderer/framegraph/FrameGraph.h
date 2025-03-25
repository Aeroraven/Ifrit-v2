
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
  String name;
  bool isImported;
  FrameGraphResourceType type;

  Ref<FgBuffer> selfBuffer;
  Ref<FgTexture> selfTexture;
  FgBuffer *importedBuffer;
  FgTexture *importedTexture;

  FgTextureSubResource subResource;
};

struct PassNode {
  PassNodeId id;
  FrameGraphPassType type;
  String name;
  bool isImported;
  Fn<void()> passFunction;
  Vec<ResourceNodeId> inputResources;
  Vec<ResourceNodeId> outputResources;
  Vec<ResourceNodeId> dependentResources;
};

class IFRIT_APIDECL FrameGraph {
private:
  Vec<ResourceNode> m_resources;
  Vec<PassNode> m_passes;

public:
  ResourceNodeId addResource(const String &name);
  PassNodeId addPass(const String &name, FrameGraphPassType type, const Vec<ResourceNodeId> &inputs,
                     const Vec<ResourceNodeId> &outputs, const Vec<ResourceNodeId> &dependencies);

  void setImportedResource(ResourceNodeId id, FgBuffer *buffer);
  void setImportedResource(ResourceNodeId id, FgTexture *texture, const FgTextureSubResource &subResource);

  void setExecutionFunction(PassNodeId id, Fn<void()> func);

  friend class FrameGraphCompiler;
  friend class FrameGraphExecutor;
};

struct CompiledFrameGraph {
  struct ResourceBarriers {
    bool enableUAVBarrier = false;
    bool enableTransitionBarrier = false;
    GraphicsBackend::Rhi::RhiResourceState srcState;
    GraphicsBackend::Rhi::RhiResourceState dstState = GraphicsBackend::Rhi::RhiResourceState::Undefined;
  };
  const FrameGraph *m_graph = nullptr;
  Vec<u32> m_passTopoOrder = {};
  Vec<Vec<ResourceBarriers>> m_passResourceBarriers = {};
  Vec<Vec<ResourceBarriers>> m_outputAliasedResourcesBarriers = {};
  Vec<Vec<u32>> m_inputResourceDependencies = {};
};

class IFRIT_APIDECL FrameGraphCompiler {
private:
public:
  CompiledFrameGraph compile(const FrameGraph &graph);
};

class IFRIT_APIDECL FrameGraphExecutor {
private:
public:
  void executeInSingleCmd(const GraphicsBackend::Rhi::RhiCommandList *cmd, const CompiledFrameGraph &compiledGraph);
};

} // namespace Ifrit::Core