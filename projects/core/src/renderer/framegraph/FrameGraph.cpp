
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

#include "ifrit/core/renderer/framegraph/FrameGraph.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/util/TypingUtil.h"
#include <stdexcept>

using Ifrit::Common::Utility::size_cast;

namespace Ifrit::Core {

IFRIT_APIDECL ResourceNodeId FrameGraph::addResource(const std::string &name) {
  ResourceNode node;
  node.id = size_cast<uint32_t>(m_resources.size());
  node.type = FrameGraphResourceType::Undefined;
  node.name = name;
  node.isImported = false;
  m_resources.push_back(node);

  // iInfo("Resource ID:{}  Name:{}", node.id, node.name);
  return node.id;
}

IFRIT_APIDECL PassNodeId FrameGraph::addPass(const std::string &name, FrameGraphPassType type,
                                             const std::vector<ResourceNodeId> &inputs,
                                             const std::vector<ResourceNodeId> &outputs,
                                             const std::vector<ResourceNodeId> &dependencies) {
  PassNode node;
  node.type = type;
  node.id = size_cast<uint32_t>(m_passes.size());
  node.name = name;
  node.isImported = false;
  node.inputResources = inputs;
  node.outputResources = outputs;
  node.dependentResources = dependencies;
  m_passes.push_back(node);
  return node.id;
}

IFRIT_APIDECL void FrameGraph::setImportedResource(ResourceNodeId id, FgBuffer *buffer) {
  m_resources[id].isImported = true;
  m_resources[id].importedBuffer = buffer;
  m_resources[id].type = FrameGraphResourceType::ResourceBuffer;
}

IFRIT_APIDECL void FrameGraph::setImportedResource(ResourceNodeId id, FgTexture *texture,
                                                   const FgTextureSubResource &subResource) {
  m_resources[id].isImported = true;
  m_resources[id].importedTexture = texture;
  m_resources[id].type = FrameGraphResourceType::ResourceTexture;
  m_resources[id].subResource = subResource;

  // iInfo("ID:{}  TextureHandle:{}", id, texture->getNativeHandle());
}

// Frame Graph compiler

// Layout transition:
// (pass)->(output:srcLayout)->(input:dstLayout)
GraphicsBackend::Rhi::RhiResourceState2 getOutputResouceState(FrameGraphPassType passType,
                                                              FrameGraphResourceType resType, FgTexture *image,
                                                              FgBuffer *buffer,
                                                              GraphicsBackend::Rhi::RhiResourceState2 parentState) {
  if (parentState != GraphicsBackend::Rhi::RhiResourceState2::Undefined) {
    return parentState;
  }
  if (resType == FrameGraphResourceType::ResourceBuffer) {
    return GraphicsBackend::Rhi::RhiResourceState2::UnorderedAccess;
  } else if (resType == FrameGraphResourceType::ResourceTexture) {
    if (passType == FrameGraphPassType::Graphics) {
      if (image->isDepthTexture()) {
        return GraphicsBackend::Rhi::RhiResourceState2::DepthStencilRT;
      } else {
        return GraphicsBackend::Rhi::RhiResourceState2::ColorRT;
      }
    } else if (passType == FrameGraphPassType::Compute) {
      // TODO: check if it's UAV or SRV
      return GraphicsBackend::Rhi::RhiResourceState2::Common;
    }
  }
  return GraphicsBackend::Rhi::RhiResourceState2::Undefined;
}
GraphicsBackend::Rhi::RhiResourceState2
getInputResourceState(FrameGraphPassType passType, FrameGraphResourceType resType, FgTexture *image, FgBuffer *buffer) {
  if (resType == FrameGraphResourceType::ResourceBuffer) {
    return GraphicsBackend::Rhi::RhiResourceState2::UnorderedAccess;
  } else if (resType == FrameGraphResourceType::ResourceTexture) {
    if (passType == FrameGraphPassType::Graphics) {
      return GraphicsBackend::Rhi::RhiResourceState2::ShaderRead;
    } else if (passType == FrameGraphPassType::Compute) {
      return GraphicsBackend::Rhi::RhiResourceState2::UnorderedAccess;
    }
  }
  return GraphicsBackend::Rhi::RhiResourceState2::Undefined;
}

IFRIT_APIDECL void FrameGraph::setExecutionFunction(PassNodeId id, std::function<void()> func) {
  m_passes[id].passFunction = func;
}

IFRIT_APIDECL CompiledFrameGraph FrameGraphCompiler::compile(const FrameGraph &graph) {
  CompiledFrameGraph compiledGraph = {};
  compiledGraph.m_inputResourceDependencies = {};
  compiledGraph.m_passTopoOrder = {};
  compiledGraph.m_passResourceBarriers = {};
  compiledGraph.m_graph = &graph;
  std::vector<uint32_t> resPassDependencies(graph.m_resources.size(), 0);
  std::vector<uint32_t> passResDepenedencies(graph.m_passes.size(), 0);
  std::vector<uint32_t> rootPasses;
  std::vector<std::vector<uint32_t>> outResToPass(graph.m_resources.size());
  std::vector<std::vector<uint32_t>> outResToPassDeps(graph.m_resources.size());

  for (const auto &pass : graph.m_passes) {
    for (auto &resId : pass.inputResources) {
      outResToPass[resId].push_back(pass.id);
    }
    for (auto &resId : pass.dependentResources) {
      outResToPassDeps[resId].push_back(pass.id);
    }
  }
  for (const auto &pass : graph.m_passes) {
    for (auto &resId : pass.outputResources) {
      resPassDependencies[resId]++;
    }
  }
  for (const auto &pass : graph.m_passes) {
    uint32_t numResToWait = 0;
    for (auto &resId : pass.inputResources) {
      if (resPassDependencies[resId] > 0) {
        numResToWait++;
      }
    }
    for (auto &resId : pass.dependentResources) {
      if (resPassDependencies[resId] > 0) {
        numResToWait++;
      }
    }
    passResDepenedencies[pass.id] = numResToWait;
    if (numResToWait == 0) {
      rootPasses.push_back(pass.id);
    }
  }
  // Resource aliasing might make incorrect resource state transition.
  // So, for each actual resource, we need to track its state
  std::unordered_map<void *, GraphicsBackend::Rhi::RhiResourceState2> rawResourceState;

  // Topological sort to arrange passes
  std::vector<GraphicsBackend::Rhi::RhiResourceState2> resState(graph.m_resources.size(),
                                                                GraphicsBackend::Rhi::RhiResourceState2::Undefined);

  // Initialize barrier vectors
  for (auto i = 0; i < graph.m_passes.size(); i++) {
    compiledGraph.m_passResourceBarriers.push_back({});
    compiledGraph.m_outputAliasedResourcesBarriers.push_back({});
  }

  while (!rootPasses.empty()) {
    auto passId0 = rootPasses.back();
    rootPasses.pop_back();
    compiledGraph.m_passTopoOrder.push_back(passId0);
    for (auto j = 0; auto &resId : graph.m_passes[passId0].outputResources) {
      void *resPtr = nullptr;
      if (graph.m_resources[resId].type == FrameGraphResourceType::ResourceBuffer) {
        resPtr = graph.m_resources[resId].importedBuffer;
      } else if (graph.m_resources[resId].type == FrameGraphResourceType::ResourceTexture) {
        resPtr = graph.m_resources[resId].importedTexture;
      }
      if (rawResourceState.find(resPtr) == rawResourceState.end()) {
        rawResourceState[resPtr] = GraphicsBackend::Rhi::RhiResourceState2::Undefined;
      }
      auto rawResState = rawResourceState[resPtr];

      resPassDependencies[resId]--;
      CompiledFrameGraph::ResourceBarriers barrier;
      compiledGraph.m_passResourceBarriers[passId0].push_back(barrier);

      auto resType = graph.m_resources[resId].type;
      auto &pass = graph.m_passes[passId0];
      auto resLayout = resState[resId];

      auto srcState = getOutputResouceState(pass.type, resType, graph.m_resources[resId].importedTexture,
                                            graph.m_resources[resId].importedBuffer, resLayout);

      CompiledFrameGraph::ResourceBarriers aliasBarrier;
      aliasBarrier.enableTransitionBarrier = false;
      aliasBarrier.enableUAVBarrier = false;
      compiledGraph.m_outputAliasedResourcesBarriers[passId0].push_back(aliasBarrier);
      if (rawResState != GraphicsBackend::Rhi::RhiResourceState2::Undefined) {
        if (srcState != rawResState) {
          // Make a transition barrier before executing the pass
          CompiledFrameGraph::ResourceBarriers aliasBarrier;
          aliasBarrier.enableTransitionBarrier = true;
          aliasBarrier.srcState = GraphicsBackend::Rhi::RhiResourceState2::AutoTraced; // rawResState;
          aliasBarrier.dstState = srcState;
          compiledGraph.m_outputAliasedResourcesBarriers[passId0].back() = aliasBarrier;
        }
      }
      // make the raw state into src state
      rawResourceState[resPtr] = srcState;

      // Output->Input barriers
      // for all subsequent passes that uses this resource
      auto dstStateAll = GraphicsBackend::Rhi::RhiResourceState2::Undefined;
      for (auto &passId : outResToPass[resId]) {
        auto dstState =
            getInputResourceState(graph.m_passes[passId].type, resType, graph.m_resources[resId].importedTexture,
                                  graph.m_resources[resId].importedBuffer);
        if (dstStateAll == GraphicsBackend::Rhi::RhiResourceState2::Undefined) {
          dstStateAll = dstState;
        } else {
          if (dstStateAll != dstState) {
            dstStateAll = GraphicsBackend::Rhi::RhiResourceState2::Common;
          }
        }
      }

      // barrier layout transitions are specified, then check if it's needed
      // if not, then it's a UAV barrier

      if (dstStateAll == GraphicsBackend::Rhi::RhiResourceState2::Undefined) {
        // no subsequent pass uses this resource
        CompiledFrameGraph::ResourceBarriers barrier;
        barrier.enableUAVBarrier = false;
        compiledGraph.m_passResourceBarriers[passId0].back() = barrier;
      } else if (srcState != dstStateAll) {
        CompiledFrameGraph::ResourceBarriers barrier;
        barrier.enableTransitionBarrier = true;
        barrier.srcState = GraphicsBackend::Rhi::RhiResourceState2::AutoTraced; // srcState;
        barrier.dstState = dstStateAll;
        compiledGraph.m_passResourceBarriers[passId0].back() = barrier;
      } else {
        CompiledFrameGraph::ResourceBarriers barrier;
        barrier.enableUAVBarrier = true;
        compiledGraph.m_passResourceBarriers[passId0].back() = barrier;
      }

      // if not undefined, make the resource state into dst state
      if (dstStateAll != GraphicsBackend::Rhi::RhiResourceState2::Undefined) {
        rawResourceState[resPtr] = dstStateAll;
      }

      if (resPassDependencies[resId] == 0) {
        for (auto &passId : outResToPass[resId]) {
          // A relation (parent)->(output)->(child) is established
          passResDepenedencies[passId]--;
          if (passResDepenedencies[passId] == 0) {
            rootPasses.push_back(passId);
          }
        }
        for (auto &passId : outResToPassDeps[resId]) {
          passResDepenedencies[passId]--;
          if (passResDepenedencies[passId] == 0) {
            rootPasses.push_back(passId);
          }
        }
      }
      j++;
    }
  }
  return compiledGraph;
}

GraphicsBackend::Rhi::RhiResourceBarrier toRhiResBarrier(const CompiledFrameGraph::ResourceBarriers &barrier,
                                                         const ResourceNode &res, bool &valid) {
  GraphicsBackend::Rhi::RhiResourceBarrier resBarrier;
  valid = false;
  if (barrier.enableTransitionBarrier) {
    resBarrier.m_type = GraphicsBackend::Rhi::RhiBarrierType::Transition;
    resBarrier.m_transition.m_type = res.type == FrameGraphResourceType::ResourceBuffer
                                         ? GraphicsBackend::Rhi::RhiResourceType::Buffer
                                         : GraphicsBackend::Rhi::RhiResourceType::Texture;
    if (res.type == FrameGraphResourceType::ResourceBuffer) {
      resBarrier.m_transition.m_buffer = res.importedBuffer;
    } else {
      resBarrier.m_transition.m_texture = res.importedTexture;
      resBarrier.m_transition.m_subResource = res.subResource;
    }
    resBarrier.m_transition.m_srcState = GraphicsBackend::Rhi::RhiResourceState2::AutoTraced; // barrier.srcState;
    resBarrier.m_transition.m_dstState = barrier.dstState;
    valid = true;
  } else if (barrier.enableUAVBarrier) {
    resBarrier.m_type = GraphicsBackend::Rhi::RhiBarrierType::UAVAccess;
    resBarrier.m_uav.m_type = res.type == FrameGraphResourceType::ResourceBuffer
                                  ? GraphicsBackend::Rhi::RhiResourceType::Buffer
                                  : GraphicsBackend::Rhi::RhiResourceType::Texture;
    if (res.type == FrameGraphResourceType::ResourceBuffer) {
      resBarrier.m_uav.m_buffer = res.importedBuffer;
    } else {
      resBarrier.m_uav.m_texture = res.importedTexture;
    }
    valid = true;
  }

  return resBarrier;
}

// Execute the compiled frame graph
IFRIT_APIDECL void FrameGraphExecutor::executeInSingleCmd(const GraphicsBackend::Rhi::RhiCommandBuffer *cmd,
                                                          const CompiledFrameGraph &compiledGraph) {

  std::vector<GraphicsBackend::Rhi::RhiResourceState2> resState(compiledGraph.m_graph->m_resources.size(),
                                                                GraphicsBackend::Rhi::RhiResourceState2::Undefined);

  for (auto &passId : compiledGraph.m_passTopoOrder) {
    std::vector<GraphicsBackend::Rhi::RhiResourceBarrier> outputAliasingBarriers;
    auto &pass = compiledGraph.m_graph->m_passes[passId];
    // For output aliased resources, make the transition barrier
    for (auto i = 0; auto &resId : pass.outputResources) {

      if (compiledGraph.m_outputAliasedResourcesBarriers[passId][i].enableTransitionBarrier) {
        bool valid = false;
        auto resBarrier = toRhiResBarrier(compiledGraph.m_outputAliasedResourcesBarriers[passId][i],
                                          compiledGraph.m_graph->m_resources[resId], valid);
        outputAliasingBarriers.push_back(resBarrier);
      }
      i++;
    }
    cmd->resourceBarrier(outputAliasingBarriers);

    // Execute the pass
    pass.passFunction();
    // Update the resource state
    std::vector<GraphicsBackend::Rhi::RhiResourceBarrier> barriers;
    for (auto i = 0; auto &resId : pass.outputResources) {
      auto srcState = compiledGraph.m_passResourceBarriers[passId][i].srcState;
      auto dstState = compiledGraph.m_passResourceBarriers[passId][i].dstState;

      if (resState[resId] != GraphicsBackend::Rhi::RhiResourceState2::Undefined && resState[resId] != srcState) {
        throw std::runtime_error("Resource state mismatch");
      }
      resState[resId] = compiledGraph.m_passResourceBarriers[passId][i].dstState;
      if (compiledGraph.m_passResourceBarriers[passId][i].enableTransitionBarrier &&
          compiledGraph.m_passResourceBarriers[passId][i].enableUAVBarrier) {
        throw std::runtime_error("Buggy design! Resource barrier can't be both transition and UAV");
      }
      bool valid = false;
      auto resBarrier = toRhiResBarrier(compiledGraph.m_passResourceBarriers[passId][i],
                                        compiledGraph.m_graph->m_resources[resId], valid);
      if (valid)
        barriers.push_back(resBarrier);
      i++;
    }
    cmd->resourceBarrier(barriers);
  }
}

} // namespace Ifrit::Core