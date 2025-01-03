
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

#include "ifrit/softgraphics/engine/base/Structures.h"
#include "ifrit/softgraphics/engine/base/VaryingDescriptor.h"
#include "ifrit/softgraphics/engine/base/VaryingStore.h"

namespace Ifrit::GraphicsBackend::SoftGraphics {
enum GeometryShaderTopology {
  IGST_TRIANGLES = 0,
  IGST_LINES = 1,
  IGST_POINTS = 2
};

class IFRIT_APIDECL ShaderBase {
public:
  float *atTexture[32];
  uint32_t atTextureWid[32];
  uint32_t atTextureHei[32];
  IfritSamplerT atSamplerPtr[32];
  char *atBuffer[32];
  bool isThreadSafe = true;
  bool forcedQuadInvocation = false;
};

class IFRIT_APIDECL VertexShader : public ShaderBase {
public:
  IFRIT_DUAL virtual void execute(const void *const *input, ifloat4 *outPos,
                                  ifloat4 *const *outVaryings) = 0;
  IFRIT_DUAL virtual ~VertexShader() = default;
  IFRIT_HOST virtual VertexShader *getCudaClone() { return nullptr; };
  IFRIT_HOST virtual std::unique_ptr<VertexShader> getThreadLocalCopy() {
    return nullptr;
  };
  IFRIT_HOST virtual void updateUniformData(int binding, int set,
                                            const void *pData) {}
  IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() {
    return {};
  }
  IFRIT_HOST virtual VaryingDescriptor getVaryingDescriptor() { return {}; }
};

class IFRIT_APIDECL FragmentShader : public ShaderBase {
public:
  bool allowDepthModification = false;
  bool requiresQuadInfo = false;
  int currentPass = 0;
  IFRIT_DUAL virtual void execute(const void *varyings, void *colorOutput,
                                  float *fragmentDepth) = 0;

  // Cuda executions are synchronized in wraps
  // So no need to call following function in cuda
  IFRIT_HOST virtual void executeInQuad(const void **varyings,
                                        void **colorOutput,
                                        float **fragmentDepth) {
    ifritError("executeInQuad not implemented");
  };

  IFRIT_DUAL virtual ~FragmentShader() = default;
  IFRIT_HOST virtual FragmentShader *getCudaClone() { return nullptr; };
  IFRIT_HOST virtual std::unique_ptr<FragmentShader> getThreadLocalCopy() {
    return nullptr;
  };
  IFRIT_HOST virtual void updateUniformData(int binding, int set,
                                            const void *pData) {}
  IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() {
    return {};
  }
};

class IFRIT_APIDECL GeometryShader : public ShaderBase {
public:
  GeometryShaderTopology atTopology = IGST_TRIANGLES;
  uint32_t atMaxVertices = 4;
  IFRIT_DUAL virtual void execute(const ifloat4 *const *inPos,
                                  const VaryingStore *const *inVaryings,
                                  ifloat4 *outPos, VaryingStore *outVaryings,
                                  int *outSize) = 0;
  IFRIT_HOST virtual GeometryShader *getCudaClone() { return nullptr; };
};

class IFRIT_APIDECL HullShader : public ShaderBase {
public:
  IFRIT_DUAL virtual void executeMain(const ifloat4 **inputPos,
                                      const VaryingStore **inputVaryings,
                                      ifloat4 *outPos,
                                      VaryingStore *outVaryings,
                                      int invocationId, int patchId) = 0;
  IFRIT_DUAL virtual void executePatchFunc(const ifloat4 **inputPos,
                                           const VaryingStore **inputVaryings,
                                           int *outerTessLevels,
                                           int *innerTessLevels,
                                           int invocationId, int patchId) = 0;
  IFRIT_HOST virtual HullShader *getCudaClone() { return nullptr; };
};

class IFRIT_APIDECL MeshShader : public ShaderBase {
public:
  IFRIT_DUAL virtual void execute(iint3 localInvocation, int workGroupId,
                                  const void *inTaskShaderPayload,
                                  VaryingStore *outVaryings, ifloat4 *outPos,
                                  int *outIndices, int &outNumVertices,
                                  int &outNumIndices) = 0;
  IFRIT_HOST virtual MeshShader *getCudaClone() { return nullptr; };
};

class IFRIT_APIDECL TaskShader : public ShaderBase {
public:
  IFRIT_DUAL virtual void execute(int workGroupId, void *outTaskShaderPayload,
                                  iint3 *outMeshWorkGroups,
                                  int &outNumMeshWorkGroups) = 0;
  IFRIT_HOST virtual TaskShader *getCudaClone() { return nullptr; };
};

/* Function Delegates & C-ABI Compatible */
typedef void (*VertexShaderFunctionalPtr)(const void *const *input,
                                          ifloat4 *outPos,
                                          ifloat4 *const *outVaryings);
typedef void (*FragmentShaderFunctionalPtr)(const void *varyings,
                                            void *colorOutput,
                                            float *fragmentDepth);
typedef void (*GeometryShaderFunctionalPtr)(const ifloat4 *const *inPos,
                                            const ifloat4 *const *inVaryings,
                                            ifloat4 *outPos,
                                            VaryingStore *outVaryings,
                                            int *outSize);
} // namespace Ifrit::GraphicsBackend::SoftGraphics