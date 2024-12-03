#pragma once
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/core/renderer/RendererBase.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <algorithm>
#include <memory>

namespace Ifrit::Core {

struct PbrAtmosphereResourceDesc {
  ifloat4 groundAlbedo;
  uint32_t atmo;
  uint32_t texTransmittance;
  uint32_t texIrradiance;
  uint32_t texScattering;
  uint32_t texMieScattering;
  float earthRadius;
  float bottomAtmoRadius;
};
struct PbrAtmospherePerframe {
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSampler;
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBuffer;

  struct PbrAtmosphereDensiyProfileLayer {
    float width;
    float expTerm;
    float expScale;   // Inv
    float linearTerm; // Inv
    float constantTerm;
    float pad0 = 114.0f;
    float pad1 = 115.0f;
    float pad2 = 116.0f;
  };

  struct PbrAtmosphereDensityProfile {
    PbrAtmosphereDensiyProfileLayer layers[2];
  };

  class PbrAtmosphereParameter {
  public:
    using vec3 = Math::SIMD::vfloat3;
    using vec4 = Math::SIMD::vfloat4;
    // some vectors here should be aligned with shader lang std
    // so, we use aligned vfloat3 here
    PbrAtmosphereDensityProfile rayleighDensity;
    PbrAtmosphereDensityProfile mieDensity;
    PbrAtmosphereDensityProfile absorptionDensity;
    float sunAngularRadius;
    float bottomRadius;
    float topRadius;
    float miePhaseFunctionG;
    float muSMin;
    float pad0;
    float pad1;
    float pad2;
    vec3 solarIrradiance;
    vec3 rayleighScattering;
    vec3 mieScattering;
    vec3 mieExtinction;
    vec3 absorptionExtinction;
    vec3 groundAlbedo;
  };
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;

  PbrAtmosphereParameter m_atmosphereParams;
  GPUBuffer *m_atmosphereParamsBuffer;

  std::shared_ptr<GPUSampler> m_sampler;
  std::shared_ptr<GPUTexture> m_transmittance;
  std::shared_ptr<GPUTexture> m_deltaIrradiance;
  std::shared_ptr<GPUTexture> m_deltaRayleighScattering;
  std::shared_ptr<GPUTexture> m_deltaMieScattering;
  std::shared_ptr<GPUTexture> m_deltaScatteringDensity;
  std::shared_ptr<GPUTexture> m_irradiance;
  std::shared_ptr<GPUTexture> m_scattering;
  std::shared_ptr<GPUTexture> m_deltaMultipleScattering;
  std::shared_ptr<GPUTexture> m_optionalSingleMieScattering;

  std::shared_ptr<GPUBindId> m_transmittanceId;
  std::shared_ptr<GPUBindId> m_deltaIrradianceId;
  std::shared_ptr<GPUBindId> m_deltaRayleighScatteringId;
  std::shared_ptr<GPUBindId> m_deltaMieScatteringId;
  std::shared_ptr<GPUBindId> m_deltaScatteringDensityId;
  std::shared_ptr<GPUBindId> m_irradianceId;
  std::shared_ptr<GPUBindId> m_scatteringId;
  std::shared_ptr<GPUBindId> m_deltaMultipleScatteringId;
  std::shared_ptr<GPUBindId> m_optionalSingleMieScatteringId;

  std::shared_ptr<GPUBindId> m_transmittanceCombSamplerId;
  std::shared_ptr<GPUBindId> m_deltaIrradianceCombSamplerId;
  std::shared_ptr<GPUBindId> m_deltaRayleighScatteringCombSamplerId;
  std::shared_ptr<GPUBindId> m_deltaMieScatteringCombSamplerId;
  std::shared_ptr<GPUBindId> m_deltaScatteringDensityCombSamplerId;
  std::shared_ptr<GPUBindId> m_irradianceCombSamplerId;
  std::shared_ptr<GPUBindId> m_scatteringCombSamplerId;
  std::shared_ptr<GPUBindId> m_deltaMultipleScatteringCombSamplerId;
  std::shared_ptr<GPUBindId> m_optionalSingleMieScatteringCombSamplerId;

  std::shared_ptr<GPUBindId> m_atmoParamId;

  float4x4 luminanceFromRad;
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
  GPUShader *createShaderFromFile(const std::string &shaderPath,
                                  const std::string &entry,
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

  virtual std::unique_ptr<GPUCommandSubmission>
  render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
         const RendererConfig &config,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::unique_ptr<GPUCommandSubmission>
  renderInternal(PerFrameData &perframe,
                 const std::vector<GPUCommandSubmission *> &cmdToWait);

  virtual PbrAtmosphereResourceDesc getResourceDesc(PerFrameData &perframe);
};

} // namespace Ifrit::Core