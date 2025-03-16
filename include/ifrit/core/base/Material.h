
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
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/core/base/AssetReference.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {
class IShaderAsset {};

enum class GraphicsShaderPassType {
  Opaque,
  Transparent,
  PostProcess,
  DirectLighting,
};

enum class ShaderEffectType { Graphics, Compute };

struct PipelineAttachmentConfigs {
  Ifrit::GraphicsBackend::Rhi::RhiImageFormat m_depthFormat;
  std::vector<Ifrit::GraphicsBackend::Rhi::RhiImageFormat> m_colorFormats;

  inline bool operator==(const PipelineAttachmentConfigs &other) const {
    auto res = m_depthFormat == other.m_depthFormat;
    res &= (m_colorFormats == other.m_colorFormats);
    return res;
  }
};

struct PipelineAttachmentConfigsHash {
  inline size_t operator()(const PipelineAttachmentConfigs &configs) const {
    size_t hash = 0;
    hash ^= std::hash<int>()(static_cast<int>(configs.m_depthFormat));
    for (const auto &format : configs.m_colorFormats) {
      hash ^= std::hash<int>()(static_cast<int>(format));
    }
    return hash;
  }
};

class ShaderEffect {
  using DrawPass = Ifrit::GraphicsBackend::Rhi::RhiGraphicsPass;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;
  using Shader = Ifrit::GraphicsBackend::Rhi::RhiShader;

public:
  ShaderEffectType m_type = ShaderEffectType::Graphics;
  std::vector<Shader *> m_shaders;
  std::vector<AssetReference> m_shaderReferences;
  std::unordered_map<PipelineAttachmentConfigs, DrawPass *, PipelineAttachmentConfigsHash> m_drawPasses;
  ComputePass *m_computePass = nullptr;

  IFRIT_STRUCT_SERIALIZE(m_shaderReferences);

  bool operator==(const ShaderEffect &other) const {
    bool result = true;
    result &= m_type == other.m_type;
    for (size_t i = 0; i < m_shaderReferences.size(); i++) {
      result &= m_shaderReferences[i] == other.m_shaderReferences[i];
    }
    return result;
  }
};

class ShaderEffectHash {
public:
  size_t operator()(const ShaderEffect &effect) const {
    size_t hash = 0;
    for (const auto &ref : effect.m_shaderReferences) {
      hash ^= std::hash<std::string>()(ref.m_uuid);
    }
    return hash;
  }
};

class Material : public IAssetCompatible {

public:
  std::string m_name;
  std::string m_uuid;
  std::vector<std::vector<char>> m_data;
  std::unordered_map<GraphicsShaderPassType, ShaderEffect> m_effectTemplates;
  std::unordered_map<GraphicsShaderPassType, std::unordered_map<std::string, u32>> m_shaderParameters;
  IFRIT_STRUCT_SERIALIZE(m_effectTemplates, m_data, m_shaderParameters);
};

} // namespace Ifrit::Core