
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

#include "ifrit/core/renderer/PbrAtmosphereRenderer.h"
#include "ifrit.shader/Atmosphere/PAS.SharedConst.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/core/util/PbrAtmoConstants.h"
#include <numbers>

namespace Ifrit::Core {

template <u32 N>
inline consteval Vector3f interpolateSpectrum(const std::array<float, N> waveLengths, const std::array<float, N> values,
                                              float scale) {
  IF_CONSTEXPR float kLamR = static_cast<float>(Util::PbrAtmoConstants::kLambdaR);
  IF_CONSTEXPR float kLamG = static_cast<float>(Util::PbrAtmoConstants::kLambdaG);
  IF_CONSTEXPR float kLamB = static_cast<float>(Util::PbrAtmoConstants::kLambdaB);
  float valR = Math::binLerp(waveLengths, values, kLamR) * scale;
  float valG = Math::binLerp(waveLengths, values, kLamG) * scale;
  float valB = Math::binLerp(waveLengths, values, kLamB) * scale;
  return {valR, valG, valB};
}

Math::SIMD::SVector3f toVec3(const Vector3f &v) { return {v.x, v.y, v.z}; }

IFRIT_APIDECL void PbrAtmosphereRenderer::preparePerframeData(PerFrameData &perframeData) {
  // todo: implement
  if (perframeData.m_atmosphereData) {
    return;
  }
  auto data = std::make_shared<PbrAtmospherePerframe>();
  perframeData.m_atmosphereData = data;

  IF_CONSTEXPR auto solarIrradiance = Util::PbrAtmoConstants::getSolarIrradiance();
  IF_CONSTEXPR auto rayleighScattering = Util::PbrAtmoConstants::getRayleighScattering();
  IF_CONSTEXPR auto mieScattering = Util::PbrAtmoConstants::getMieScattering();
  IF_CONSTEXPR auto mieExtinction = Util::PbrAtmoConstants::getMieExtinction();
  IF_CONSTEXPR auto absorptionExtinction = Util::PbrAtmoConstants::getAbsorptionExtinction();

  IF_CONSTEXPR auto km = 1e3;
  IF_CONSTEXPR auto kmX = 1e0;
  IF_CONSTEXPR auto degToRad = std::numbers::pi_v<float> / 180.0f;
  IF_CONSTEXPR auto sunAngularRadius = 0.2678 * degToRad;
  IF_CONSTEXPR auto bottomRadius = 6360.0 * km;
  IF_CONSTEXPR auto topRadius = 6420.0 * km;

  using DensityProfileLayer = PbrAtmospherePerframe::PbrAtmosphereDensiyProfileLayer;
  IF_CONSTEXPR DensityProfileLayer rayleighLayer1 = {
      0.0f, 1.0f, static_cast<float>(-1.0 / Util::PbrAtmoConstants::kRayleighScaleHeight * km), 0.0, 0.0f};
  IF_CONSTEXPR DensityProfileLayer mieLayer1 = {
      0.0f, 1.0f, static_cast<float>(-1.0 / Util::PbrAtmoConstants::kMieScaleHeight * km), 0.0f, 0.0f};

  IF_CONSTEXPR DensityProfileLayer absorptionLayer0 = {static_cast<float>(25.0 * kmX), 0.0, 0.0,
                                                       static_cast<float>(1.0 / (15.0 * kmX)), -2.0f / 3.0f};
  IF_CONSTEXPR DensityProfileLayer absorptionLayer1 = {0.0f, 0.0f, 0.0f, static_cast<float>(-1.0 / (15.0 * kmX)),
                                                       8.0f / 3.0f};
  IF_CONSTEXPR auto muSMin = gcem::cos(102.0 * degToRad);

  IF_CONSTEXPR Vector3f earthCenter = {0.0f, 0.0f, static_cast<float>(-bottomRadius)};
  IF_CONSTEXPR Vector3f groundAlbedo = {0.1f, 0.1f, 0.1f};
  IF_CONSTEXPR Vector2f sunSize = {static_cast<float>(gcem::tan(sunAngularRadius)),
                                   static_cast<float>(gcem::cos(sunAngularRadius))};

  // Now filling in the data (params)
  IF_CONSTEXPR auto solarIrradianceFloat = Math::convertArray<double, float, solarIrradiance.size()>(solarIrradiance);
  IF_CONSTEXPR auto waveLengthToSample = Math::uniformSampleIncl<float, solarIrradiance.size()>(
      Util::PbrAtmoConstants::kLambdaMin, Util::PbrAtmoConstants::kLambdaMax);

  data->m_atmosphereParams.solarIrradiance =
      toVec3(interpolateSpectrum(waveLengthToSample, solarIrradianceFloat, 1.0f));
  data->m_atmosphereParams.sunAngularRadius = static_cast<float>(sunAngularRadius);
  data->m_atmosphereParams.bottomRadius = bottomRadius / Util::PbrAtmoConstants::km;
  data->m_atmosphereParams.topRadius = topRadius / Util::PbrAtmoConstants::km;
  data->m_atmosphereParams.rayleighDensity.layers[0] = rayleighLayer1;
  data->m_atmosphereParams.rayleighDensity.layers[1] = rayleighLayer1;
  data->m_atmosphereParams.rayleighScattering =
      toVec3(interpolateSpectrum(waveLengthToSample, rayleighScattering, Util::PbrAtmoConstants::km));

  data->m_atmosphereParams.mieDensity.layers[0] = mieLayer1;
  data->m_atmosphereParams.mieDensity.layers[1] = mieLayer1;
  data->m_atmosphereParams.mieScattering =
      toVec3(interpolateSpectrum(waveLengthToSample, mieScattering, Util::PbrAtmoConstants::km));

  data->m_atmosphereParams.mieExtinction =
      toVec3(interpolateSpectrum(waveLengthToSample, mieExtinction, Util::PbrAtmoConstants::km));

  data->m_atmosphereParams.miePhaseFunctionG = static_cast<float>(Util::PbrAtmoConstants::kMiePhaseFunctionG);
  data->m_atmosphereParams.absorptionDensity.layers[0] = absorptionLayer0;
  data->m_atmosphereParams.absorptionDensity.layers[1] = absorptionLayer1;

  data->m_atmosphereParams.absorptionExtinction =
      toVec3(interpolateSpectrum(waveLengthToSample, absorptionExtinction, Util::PbrAtmoConstants::km));

  data->m_atmosphereParams.groundAlbedo = toVec3(groundAlbedo);
  data->m_atmosphereParams.muSMin = static_cast<float>(muSMin);

  // create textures
  auto rhi = m_app->getRhiLayer();
  auto createPbrAtmoTex2D = [&](u32 width, u32 height) {
    using namespace GraphicsBackend::Rhi;
    auto tex =
        rhi->createTexture2D("PbrAtmo_Tex2D", width, height, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                             RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT |
                                 RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT,
                             true);
    return tex;
  };
  auto createPbrAtmoTex3D = [&](u32 width, u32 height, u32 depth) {
    using namespace GraphicsBackend::Rhi;
    auto tex =
        rhi->createTexture3D("PbrAtmo_Tex3D", width, height, depth, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                             RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT |
                                 RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT,
                             true);
    return tex;
  };
  data->m_sampler = rhi->createTrivialSampler();
  data->m_transmittance = createPbrAtmoTex2D(AtmospherePASConfig::TRANSMITTANCE_TEXTURE_WIDTH,
                                             AtmospherePASConfig::TRANSMITTANCE_TEXTURE_HEIGHT);
  data->m_deltaIrradiance =
      createPbrAtmoTex2D(AtmospherePASConfig::IRRADIANCE_TEXTURE_WIDTH, AtmospherePASConfig::IRRADIANCE_TEXTURE_HEIGHT);
  data->m_irradiance =
      createPbrAtmoTex2D(AtmospherePASConfig::IRRADIANCE_TEXTURE_WIDTH, AtmospherePASConfig::IRRADIANCE_TEXTURE_HEIGHT);
  data->m_scattering =
      createPbrAtmoTex3D(AtmospherePASConfig::SCATTERING_TEXTURE_WIDTH, AtmospherePASConfig::SCATTERING_TEXTURE_HEIGHT,
                         AtmospherePASConfig::SCATTERING_TEXTURE_DEPTH);
  data->m_optionalSingleMieScattering =
      createPbrAtmoTex3D(AtmospherePASConfig::SCATTERING_TEXTURE_WIDTH, AtmospherePASConfig::SCATTERING_TEXTURE_HEIGHT,
                         AtmospherePASConfig::SCATTERING_TEXTURE_DEPTH);
  data->m_deltaRayleighScattering =
      createPbrAtmoTex3D(AtmospherePASConfig::SCATTERING_TEXTURE_WIDTH, AtmospherePASConfig::SCATTERING_TEXTURE_HEIGHT,
                         AtmospherePASConfig::SCATTERING_TEXTURE_DEPTH);
  data->m_deltaMieScattering =
      createPbrAtmoTex3D(AtmospherePASConfig::SCATTERING_TEXTURE_WIDTH, AtmospherePASConfig::SCATTERING_TEXTURE_HEIGHT,
                         AtmospherePASConfig::SCATTERING_TEXTURE_DEPTH);
  data->m_deltaScatteringDensity =
      createPbrAtmoTex3D(AtmospherePASConfig::SCATTERING_TEXTURE_WIDTH, AtmospherePASConfig::SCATTERING_TEXTURE_HEIGHT,
                         AtmospherePASConfig::SCATTERING_TEXTURE_DEPTH);
  data->m_deltaMultipleScattering = data->m_deltaRayleighScattering;

  // bindless ids for combined image sampler
  data->m_transmittanceCombSamplerId =
      rhi->registerCombinedImageSampler(data->m_transmittance.get(), data->m_sampler.get());
  data->m_deltaIrradianceCombSamplerId =
      rhi->registerCombinedImageSampler(data->m_deltaIrradiance.get(), data->m_sampler.get());
  data->m_irradianceCombSamplerId = rhi->registerCombinedImageSampler(data->m_irradiance.get(), data->m_sampler.get());
  data->m_scatteringCombSamplerId = rhi->registerCombinedImageSampler(data->m_scattering.get(), data->m_sampler.get());
  data->m_optionalSingleMieScatteringCombSamplerId =
      rhi->registerCombinedImageSampler(data->m_optionalSingleMieScattering.get(), data->m_sampler.get());
  data->m_deltaRayleighScatteringCombSamplerId =
      rhi->registerCombinedImageSampler(data->m_deltaRayleighScattering.get(), data->m_sampler.get());
  data->m_deltaMieScatteringCombSamplerId =
      rhi->registerCombinedImageSampler(data->m_deltaMieScattering.get(), data->m_sampler.get());
  data->m_deltaScatteringDensityCombSamplerId =
      rhi->registerCombinedImageSampler(data->m_deltaScatteringDensity.get(), data->m_sampler.get());
  data->m_deltaMultipleScatteringCombSamplerId = data->m_deltaRayleighScatteringCombSamplerId;

  // Copy atmo params to GPU
  data->m_atmosphereParamsBuffer =
      rhi->createBufferDevice("PbrAtmo_Params", sizeof(PbrAtmospherePerframe::PbrAtmosphereParameter),
                              GraphicsBackend::Rhi::RhiBufferUsage::RhiBufferUsage_CopyDst |
                                  GraphicsBackend::Rhi::RhiBufferUsage::RhiBufferUsage_SSBO,
                              true);
  auto stagingBuffer = rhi->createStagedSingleBuffer(data->m_atmosphereParamsBuffer.get());
  auto tq = rhi->getQueue(GraphicsBackend::Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
  tq->runSyncCommand([&](const GraphicsBackend::Rhi::RhiCommandList *cmd) {
    stagingBuffer->cmdCopyToDevice(cmd, &data->m_atmosphereParams,
                                   sizeof(PbrAtmospherePerframe::PbrAtmosphereParameter), 0);
  });
  // Last, a matrix to convert radiance to luminance
  data->luminanceFromRad = Math::identity4();
}

IFRIT_APIDECL PbrAtmosphereRenderer::GPUShader *
PbrAtmosphereRenderer::createShaderFromFile(const std::string &shaderPath, const std::string &entry,
                                            GraphicsBackend::Rhi::RhiShaderStage stage) {
  auto rhi = m_app->getRhiLayer();
  std::string shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  auto path = shaderBasePath + "/atmosphere/" + shaderPath;
  auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
  std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  return rhi->createShader(path, shaderCodeVec, entry, stage, GraphicsBackend::Rhi::RhiShaderSourceType::GLSLCode);
}

IFRIT_APIDECL void PbrAtmosphereRenderer::setupTransmittancePrecomputePass() {
  auto rhi = m_app->getRhiLayer();
  auto shader =
      createShaderFromFile("PAS.ComputeTransmittance.comp.glsl", "main", GraphicsBackend::Rhi::RhiShaderStage::Compute);
  m_transmittancePrecomputePass = rhi->createComputePass();
  m_transmittancePrecomputePass->setComputeShader(shader);
  m_transmittancePrecomputePass->setNumBindlessDescriptorSets(0);
  m_transmittancePrecomputePass->setPushConstSize(sizeof(u32) * 2);
}

IFRIT_APIDECL void PbrAtmosphereRenderer::setupIrradiancePrecomputePass() {
  auto rhi = m_app->getRhiLayer();
  auto shader =
      createShaderFromFile("PAS.ComputeIrradiance.comp.glsl", "main", GraphicsBackend::Rhi::RhiShaderStage::Compute);
  m_irradiancePrecomputePass = rhi->createComputePass();
  m_irradiancePrecomputePass->setComputeShader(shader);
  m_irradiancePrecomputePass->setNumBindlessDescriptorSets(0);
  m_irradiancePrecomputePass->setPushConstSize(sizeof(u32) * 4);
}

IFRIT_APIDECL void PbrAtmosphereRenderer::setupSingleScatteringPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("PAS.ComputeSingleScattering.comp.glsl", "main",
                                     GraphicsBackend::Rhi::RhiShaderStage::Compute);
  m_singleScatteringPass = rhi->createComputePass();
  m_singleScatteringPass->setComputeShader(shader);
  m_singleScatteringPass->setNumBindlessDescriptorSets(0);
  m_singleScatteringPass->setPushConstSize(sizeof(u32) * 6 + sizeof(Matrix4x4f));
}

IFRIT_APIDECL void PbrAtmosphereRenderer::setupScatteringDensityPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("PAS.ComputeScatteringDensity.comp.glsl", "main",
                                     GraphicsBackend::Rhi::RhiShaderStage::Compute);
  m_scatteringDensity = rhi->createComputePass();
  m_scatteringDensity->setComputeShader(shader);
  m_scatteringDensity->setNumBindlessDescriptorSets(0);
  m_scatteringDensity->setPushConstSize(sizeof(u32) * 8);
}

IFRIT_APIDECL void PbrAtmosphereRenderer::setupIndirectIrradiancePass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("PAS.ComputeIndirectIrradiance.comp.glsl", "main",
                                     GraphicsBackend::Rhi::RhiShaderStage::Compute);
  m_indirectIrradiancePass = rhi->createComputePass();
  m_indirectIrradiancePass->setComputeShader(shader);
  m_indirectIrradiancePass->setNumBindlessDescriptorSets(0);
  m_indirectIrradiancePass->setPushConstSize(sizeof(u32) * 7 + sizeof(Matrix4x4f));
}

IFRIT_APIDECL void PbrAtmosphereRenderer::setupMultipleScatteringPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("PAS.ComputeMultipleScattering.comp.glsl", "main",
                                     GraphicsBackend::Rhi::RhiShaderStage::Compute);
  m_multipleScatteringPass = rhi->createComputePass();
  m_multipleScatteringPass->setComputeShader(shader);
  m_multipleScatteringPass->setNumBindlessDescriptorSets(0);
  m_multipleScatteringPass->setPushConstSize(sizeof(u32) * 5 + sizeof(Matrix4x4f));
}

IFRIT_APIDECL std::unique_ptr<PbrAtmosphereRenderer::GPUCommandSubmission>
PbrAtmosphereRenderer::renderInternal(PerFrameData &perframe, const std::vector<GPUCommandSubmission *> &cmdToWait) {
  using namespace Ifrit::Math;
  using namespace GraphicsBackend::Rhi;

  preparePerframeData(perframe);
  // Step1. precompute transmittance
  struct PushConstTransmittanceStruct {
    u32 atmoData;
    u32 transmittanceRef;
  } pcTransmittance;
  auto data = reinterpret_cast<PbrAtmospherePerframe *>(perframe.m_atmosphereData.get());
  pcTransmittance.atmoData = data->m_atmosphereParamsBuffer->getDescId();
  pcTransmittance.transmittanceRef = data->m_transmittance->getDescId();
  m_transmittancePrecomputePass->setRecordFunction([&](RhiRenderPassContext *ctx) {
    ctx->m_cmd->setPushConst(m_transmittancePrecomputePass, 0, sizeof(pcTransmittance), &pcTransmittance);
    IF_CONSTEXPR auto wgX =
        divRoundUp(AtmospherePASConfig::TRANSMITTANCE_TEXTURE_WIDTH, AtmospherePASConfig::cPasTransmittanceTGX);
    IF_CONSTEXPR auto wgY =
        divRoundUp(AtmospherePASConfig::TRANSMITTANCE_TEXTURE_HEIGHT, AtmospherePASConfig::cPasTransmittanceTGY);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });
  // An UAV barrier is required for the transmittance texture
  auto createUAVImageBarrier = [&](RhiTexture *img) {
    RhiResourceBarrier barrier;
    barrier.m_uav.m_texture = img;
    barrier.m_uav.m_type = RhiResourceType::Texture;
    barrier.m_type = RhiBarrierType::UAVAccess;
    return barrier;
  };
  auto transmittanceBarrier = createUAVImageBarrier(data->m_transmittance.get());

  // Step2. precompute irradiance
  struct PushConstIrradianceStruct {
    u32 atmoData;
    u32 transmittanceRef;
    u32 irradianceRef;
    u32 deltaIrradianceRef;
  } pcIrradiance;

  pcIrradiance.atmoData = data->m_atmosphereParamsBuffer->getDescId();
  pcIrradiance.transmittanceRef = data->m_transmittanceCombSamplerId->getActiveId();
  pcIrradiance.irradianceRef = data->m_irradiance->getDescId();
  pcIrradiance.deltaIrradianceRef = data->m_deltaIrradiance->getDescId();
  m_irradiancePrecomputePass->setRecordFunction([&](RhiRenderPassContext *ctx) {
    ctx->m_cmd->setPushConst(m_irradiancePrecomputePass, 0, sizeof(pcIrradiance), &pcIrradiance);
    IF_CONSTEXPR auto wgX =
        divRoundUp(AtmospherePASConfig::IRRADIANCE_TEXTURE_WIDTH, AtmospherePASConfig::cPasIrradianceTGX);
    IF_CONSTEXPR auto wgY =
        divRoundUp(AtmospherePASConfig::IRRADIANCE_TEXTURE_HEIGHT, AtmospherePASConfig::cPasIrradianceTGY);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });

  // Step3. precompute single scattering
  struct PushConstSingleScatteringStruct {
    Matrix4x4f lumFromRad;
    u32 atmoData;
    u32 deltaRayleigh;
    u32 deltaMie;
    u32 scattering;
    u32 singleMieScattering;
    u32 transmittanceSampler;
  } pSingleScattering;

  pSingleScattering.lumFromRad = Math::identity4();
  pSingleScattering.atmoData = data->m_atmosphereParamsBuffer->getDescId();
  pSingleScattering.deltaRayleigh = data->m_deltaRayleighScattering->getDescId();
  pSingleScattering.deltaMie = data->m_deltaMieScattering->getDescId();
  pSingleScattering.scattering = data->m_scattering->getDescId();
  pSingleScattering.singleMieScattering = data->m_optionalSingleMieScattering->getDescId();
  pSingleScattering.transmittanceSampler = data->m_transmittanceCombSamplerId->getActiveId();
  m_singleScatteringPass->setRecordFunction([&](RhiRenderPassContext *ctx) {
    ctx->m_cmd->setPushConst(m_singleScatteringPass, 0, sizeof(pSingleScattering), &pSingleScattering);
    IF_CONSTEXPR auto wgX =
        divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_WIDTH, AtmospherePASConfig::cPasSingleScatteringTGX);
    IF_CONSTEXPR auto wgY =
        divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_HEIGHT, AtmospherePASConfig::cPasSingleScatteringTGY);
    IF_CONSTEXPR auto wgZ =
        divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_DEPTH, AtmospherePASConfig::cPasSingleScatteringTGZ);
    ctx->m_cmd->dispatch(wgX, wgY, wgZ);
  });

  // Step4. For higher order scattering
  auto singleRayleighScatterBarrier = createUAVImageBarrier(data->m_deltaRayleighScattering.get());
  auto singleMieScatterBarrier = createUAVImageBarrier(data->m_deltaMieScattering.get());
  auto scatteringBarrier = createUAVImageBarrier(data->m_scattering.get());
  auto irradianceBarrier = createUAVImageBarrier(data->m_irradiance.get());
  auto deltaIrradianceBarrier = createUAVImageBarrier(data->m_deltaIrradiance.get());
  auto multipleScatteringBarrier = createUAVImageBarrier(data->m_deltaMultipleScattering.get());
  auto scatterDensityBarrier = createUAVImageBarrier(data->m_deltaScatteringDensity.get());

  struct PushConstScatteringDensityStruct {
    u32 atmoData;
    u32 transmittanceSampler;
    u32 singleRayleighScatterSampler;
    u32 singleMieScatterSampler;
    u32 multipleScatteringSampler;
    u32 irradianceSampler;
    u32 scatterDensity;
    u32 scatterOrder;
  } pScatteringDensity;

  struct PushConstIndirectIrradianceStruct {
    Matrix4x4f lumFromRad;
    u32 atmoData;
    u32 deltaIrradiance;
    u32 irradiance;
    u32 singleRayleighScatteringSamp;
    u32 singleMieScatteringSamp;
    u32 multipleScatteringSamp;
    u32 scatteringOrder;
  } pIndirectIrradiance;

  struct PushConstMultipleScatteringStruct {
    Matrix4x4f lumFromRad;
    u32 atmoData;
    u32 deltaMultipleScattering;
    u32 scattering;
    u32 transmittanceSamp;
    u32 scatteringDensitySamp;
  } pMultipleScattering;

  auto recordCmdOrder = [&](u32 order) {
    pScatteringDensity.atmoData = data->m_atmosphereParamsBuffer->getDescId();
    pScatteringDensity.transmittanceSampler = data->m_transmittanceCombSamplerId->getActiveId();
    pScatteringDensity.singleRayleighScatterSampler = data->m_deltaRayleighScatteringCombSamplerId->getActiveId();
    pScatteringDensity.singleMieScatterSampler = data->m_deltaMieScatteringCombSamplerId->getActiveId();
    pScatteringDensity.multipleScatteringSampler = data->m_deltaMultipleScatteringCombSamplerId->getActiveId();
    pScatteringDensity.irradianceSampler = data->m_deltaIrradianceCombSamplerId->getActiveId();
    pScatteringDensity.scatterDensity = data->m_deltaScatteringDensity->getDescId();
    pScatteringDensity.scatterOrder = order;

    pIndirectIrradiance.lumFromRad = Math::identity4();
    pIndirectIrradiance.atmoData = data->m_atmosphereParamsBuffer->getDescId();
    pIndirectIrradiance.deltaIrradiance = data->m_deltaIrradiance->getDescId();
    pIndirectIrradiance.irradiance = data->m_irradiance->getDescId();
    pIndirectIrradiance.singleRayleighScatteringSamp = data->m_deltaRayleighScatteringCombSamplerId->getActiveId();
    pIndirectIrradiance.singleMieScatteringSamp = data->m_deltaMieScatteringCombSamplerId->getActiveId();
    pIndirectIrradiance.multipleScatteringSamp = data->m_deltaMultipleScatteringCombSamplerId->getActiveId();
    pIndirectIrradiance.scatteringOrder = order - 1;

    pMultipleScattering.lumFromRad = Math::identity4();
    pMultipleScattering.atmoData = data->m_atmosphereParamsBuffer->getDescId();
    pMultipleScattering.deltaMultipleScattering = data->m_deltaMultipleScattering->getDescId();
    pMultipleScattering.scattering = data->m_scattering->getDescId();
    pMultipleScattering.transmittanceSamp = data->m_transmittanceCombSamplerId->getActiveId();
    pMultipleScattering.scatteringDensitySamp = data->m_deltaScatteringDensityCombSamplerId->getActiveId();

    m_scatteringDensity->setRecordFunction([&](RhiRenderPassContext *ctx) {
      ctx->m_cmd->setPushConst(m_scatteringDensity, 0, sizeof(pScatteringDensity), &pScatteringDensity);
      IF_CONSTEXPR auto wgX =
          divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_WIDTH, AtmospherePASConfig::cPasScatteringDensityTGX);
      IF_CONSTEXPR auto wgY =
          divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_HEIGHT, AtmospherePASConfig::cPasScatteringDensityTGY);
      IF_CONSTEXPR auto wgZ =
          divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_DEPTH, AtmospherePASConfig::cPasScatteringDensityTGZ);
      ctx->m_cmd->dispatch(wgX, wgY, wgZ);
    });

    m_indirectIrradiancePass->setRecordFunction([&](RhiRenderPassContext *ctx) {
      ctx->m_cmd->setPushConst(m_indirectIrradiancePass, 0, sizeof(pIndirectIrradiance), &pIndirectIrradiance);
      IF_CONSTEXPR auto wgX =
          divRoundUp(AtmospherePASConfig::IRRADIANCE_TEXTURE_WIDTH, AtmospherePASConfig::cPasIndirectIrradianceTGX);
      IF_CONSTEXPR auto wgY =
          divRoundUp(AtmospherePASConfig::IRRADIANCE_TEXTURE_HEIGHT, AtmospherePASConfig::cPasIndirectIrradianceTGY);
      ctx->m_cmd->dispatch(wgX, wgY, 1);
    });

    m_multipleScatteringPass->setRecordFunction([&](RhiRenderPassContext *ctx) {
      ctx->m_cmd->setPushConst(m_multipleScatteringPass, 0, sizeof(pMultipleScattering), &pMultipleScattering);
      IF_CONSTEXPR auto wgX =
          divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_WIDTH, AtmospherePASConfig::cPasMultipleScatteringTGX);
      IF_CONSTEXPR auto wgY =
          divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_HEIGHT, AtmospherePASConfig::cPasMultipleScatteringTGY);
      IF_CONSTEXPR auto wgZ =
          divRoundUp(AtmospherePASConfig::SCATTERING_TEXTURE_DEPTH, AtmospherePASConfig::cPasMultipleScatteringTGZ);
      ctx->m_cmd->dispatch(wgX, wgY, wgZ);
    });
  };

  auto runCmdOrder = [&](const RhiCommandList *cmd, u32 order) {
    recordCmdOrder(order);
    cmd->resourceBarrier({singleRayleighScatterBarrier, singleMieScatterBarrier, scatteringBarrier, irradianceBarrier,
                          deltaIrradianceBarrier, multipleScatteringBarrier});
    m_scatteringDensity->run(cmd, 0);
    cmd->resourceBarrier({scatterDensityBarrier, irradianceBarrier, deltaIrradianceBarrier});
    m_indirectIrradiancePass->run(cmd, 0);
    cmd->resourceBarrier({irradianceBarrier, deltaIrradianceBarrier});
    m_multipleScatteringPass->run(cmd, 0);
    cmd->resourceBarrier({multipleScatteringBarrier, scatteringBarrier});
  };

  // Final, run
  auto cq = m_app->getRhiLayer()->getQueue(GraphicsBackend::Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);

  auto toGeneralLayout = [&](const RhiCommandList *cmd, RhiTexture *tex) {
    RhiTransitionBarrier tBarrier;
    tBarrier.m_texture = tex;
    tBarrier.m_type = RhiResourceType::Texture;
    tBarrier.m_dstState = RhiResourceState::Common;
    tBarrier.m_subResource = {0, 0, 1, 1};

    RhiResourceBarrier barrier;
    barrier.m_type = RhiBarrierType::Transition;
    barrier.m_transition = tBarrier;

    cmd->resourceBarrier({barrier});
  };
  auto task = cq->runAsyncCommand(
      [&](const RhiCommandList *cmd) {
        cmd->beginScope("Precompute Atmosphere Scattering");
        toGeneralLayout(cmd, data->m_transmittance.get());
        toGeneralLayout(cmd, data->m_deltaIrradiance.get());
        toGeneralLayout(cmd, data->m_irradiance.get());
        toGeneralLayout(cmd, data->m_scattering.get());
        toGeneralLayout(cmd, data->m_optionalSingleMieScattering.get());
        toGeneralLayout(cmd, data->m_deltaRayleighScattering.get());
        toGeneralLayout(cmd, data->m_deltaMieScattering.get());
        toGeneralLayout(cmd, data->m_deltaScatteringDensity.get());
        toGeneralLayout(cmd, data->m_deltaMultipleScattering.get());

        m_transmittancePrecomputePass->run(cmd, 0);
        cmd->resourceBarrier({transmittanceBarrier});
        m_irradiancePrecomputePass->run(cmd, 0);
        m_singleScatteringPass->run(cmd, 0);
        for (u32 i = 2; i <= 4; i++) {
          runCmdOrder(cmd, i);
        }
        cmd->endScope();
      },
      cmdToWait, {});
  return task;
}

IFRIT_APIDECL PbrAtmosphereResourceDesc PbrAtmosphereRenderer::getResourceDesc(PerFrameData &perframe) {
  PbrAtmosphereResourceDesc desc;
  auto data = reinterpret_cast<PbrAtmospherePerframe *>(perframe.m_atmosphereData.get());
  desc.atmo = data->m_atmosphereParamsBuffer->getDescId();
  desc.texIrradiance = data->m_irradianceCombSamplerId->getActiveId();
  desc.texMieScattering = data->m_optionalSingleMieScatteringCombSamplerId->getActiveId();
  desc.texScattering = data->m_scatteringCombSamplerId->getActiveId();
  desc.texTransmittance = data->m_transmittanceCombSamplerId->getActiveId();
  desc.earthRadius = data->m_atmosphereParams.bottomRadius;
  desc.bottomAtmoRadius = data->m_atmosphereParams.bottomRadius;
  desc.groundAlbedo = Vector4f(0.1f, 0.1f, 0.1f, 1.0f);
  return desc;
}

} // namespace Ifrit::Core