
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
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/core/renderer/RendererBase.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <algorithm>
#include <memory>

namespace Ifrit::Core {

struct PbrAtmosphereResourceDesc {
  Vector4f groundAlbedo;
  u32 atmo;
  u32 texTransmittance;
  u32 texIrradiance;
  u32 texScattering;
  u32 texMieScattering;
  f32 earthRadius;
  f32 bottomAtmoRadius;
};
struct PbrAtmospherePerframe {
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTextureRef;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSamplerRef;
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBufferRef;

  struct PbrAtmosphereDensiyProfileLayer {
    f32 width;
    f32 expTerm;
    f32 expScale;   // Inv
    f32 linearTerm; // Inv
    f32 constantTerm;
    f32 pad0 = 114.0f;
    f32 pad1 = 115.0f;
    f32 pad2 = 116.0f;
  };

  struct PbrAtmosphereDensityProfile {
    PbrAtmosphereDensiyProfileLayer layers[2];
  };

  class PbrAtmosphereParameter {
  public:
    using vec3 = Math::SIMD::SVector3f;
    using vec4 = Math::SIMD::SVector4f;
    // some vectors here should be aligned with shader lang std
    // so, we use aligned SVector3f here
    PbrAtmosphereDensityProfile rayleighDensity;
    PbrAtmosphereDensityProfile mieDensity;
    PbrAtmosphereDensityProfile absorptionDensity;
    f32 sunAngularRadius;
    f32 bottomRadius;
    f32 topRadius;
    f32 miePhaseFunctionG;
    f32 muSMin;
    f32 pad0;
    f32 pad1;
    f32 pad2;
    vec3 solarIrradiance;
    vec3 rayleighScattering;
    vec3 mieScattering;
    vec3 mieExtinction;
    vec3 absorptionExtinction;
    vec3 groundAlbedo;
  };
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiDescHandleLegacy;

  PbrAtmosphereParameter m_atmosphereParams;
  GPUBuffer m_atmosphereParamsBuffer;

  GPUSampler m_sampler;
  GPUTexture m_transmittance;
  GPUTexture m_deltaIrradiance;
  GPUTexture m_deltaRayleighScattering;
  GPUTexture m_deltaMieScattering;
  GPUTexture m_deltaScatteringDensity;
  GPUTexture m_irradiance;
  GPUTexture m_scattering;
  GPUTexture m_deltaMultipleScattering;
  GPUTexture m_optionalSingleMieScattering;

  // Ref<GPUBindId> m_transmittanceId;
  // Ref<GPUBindId> m_deltaIrradianceId;
  // Ref<GPUBindId> m_deltaRayleighScatteringId;
  // Ref<GPUBindId> m_deltaMieScatteringId;
  // Ref<GPUBindId> m_deltaScatteringDensityId;
  // Ref<GPUBindId> m_irradianceId;
  // Ref<GPUBindId> m_scatteringId;
  // Ref<GPUBindId> m_deltaMultipleScatteringId;
  // Ref<GPUBindId> m_optionalSingleMieScatteringId;

  Ref<GPUBindId> m_transmittanceCombSamplerId;
  Ref<GPUBindId> m_deltaIrradianceCombSamplerId;
  Ref<GPUBindId> m_deltaRayleighScatteringCombSamplerId;
  Ref<GPUBindId> m_deltaMieScatteringCombSamplerId;
  Ref<GPUBindId> m_deltaScatteringDensityCombSamplerId;
  Ref<GPUBindId> m_irradianceCombSamplerId;
  Ref<GPUBindId> m_scatteringCombSamplerId;
  Ref<GPUBindId> m_deltaMultipleScatteringCombSamplerId;
  Ref<GPUBindId> m_optionalSingleMieScatteringCombSamplerId;

  Matrix4x4f luminanceFromRad;
};

class IFRIT_APIDECL PbrAtmosphereRenderer : public RendererBase {
  using GPUCommandSubmission = Ifrit::GraphicsBackend::Rhi::RhiTaskSubmission;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUShader = Ifrit::GraphicsBackend::Rhi::RhiShader;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;

private:
  ComputePass *m_transmittancePrecomputePass;
  ComputePass *m_irradiancePrecomputePass;
  ComputePass *m_singleScatteringPass;
  ComputePass *m_scatteringDensity;
  ComputePass *m_indirectIrradiancePass;
  ComputePass *m_multipleScatteringPass;

protected:
  void setupTransmittancePrecomputePass();
  void setupIrradiancePrecomputePass();
  void setupSingleScatteringPass();
  void setupScatteringDensityPass();
  void setupIndirectIrradiancePass();
  void setupMultipleScatteringPass();

  void preparePerframeData(PerFrameData &perframeData);
  GPUShader *createShaderFromFile(const std::string &shaderPath, const std::string &entry,
                                  GraphicsBackend::Rhi::RhiShaderStage stage);

public:
  PbrAtmosphereRenderer(IApplication *app) : RendererBase(app) {
    setupTransmittancePrecomputePass();
    setupIrradiancePrecomputePass();
    setupSingleScatteringPass();
    setupScatteringDensityPass();
    setupIndirectIrradiancePass();
    setupMultipleScatteringPass();
  }

  virtual std::unique_ptr<GPUCommandSubmission> render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
                                                       const RendererConfig &config,
                                                       const std::vector<GPUCommandSubmission *> &cmdToWait) override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::unique_ptr<GPUCommandSubmission> renderInternal(PerFrameData &perframe,
                                                               const std::vector<GPUCommandSubmission *> &cmdToWait);

  virtual PbrAtmosphereResourceDesc getResourceDesc(PerFrameData &perframe);
};

} // namespace Ifrit::Core