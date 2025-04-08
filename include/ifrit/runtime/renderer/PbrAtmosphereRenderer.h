
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/simd/SimdVectors.h"
#include "ifrit/runtime/renderer/RendererBase.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <algorithm>
#include <memory>

namespace Ifrit::Runtime
{

    struct PbrAtmosphereResourceDesc
    {
        Vector4f groundAlbedo;
        u32      atmo;
        u32      texTransmittance;
        u32      texIrradiance;
        u32      texScattering;
        u32      texMieScattering;
        f32      earthRadius;
        f32      bottomAtmoRadius;
    };
    struct PbrAtmospherePerframe
    {
        using GPUTexture = Graphics::Rhi::RhiTextureRef;
        using GPUBuffer  = Graphics::Rhi::RhiBufferRef;

        struct PbrAtmosphereDensiyProfileLayer
        {
            f32 width;
            f32 expTerm;
            f32 expScale;   // Inv
            f32 linearTerm; // Inv
            f32 constantTerm;
            f32 pad0 = 114.0f;
            f32 pad1 = 115.0f;
            f32 pad2 = 116.0f;
        };

        struct PbrAtmosphereDensityProfile
        {
            PbrAtmosphereDensiyProfileLayer layers[2];
        };

        class PbrAtmosphereParameter
        {
        public:
            using vec3 = Math::SIMD::SVector3f;
            using vec4 = Math::SIMD::SVector4f;
            // some vectors here should be aligned with shader lang std
            // so, we use aligned SVector3f here
            PbrAtmosphereDensityProfile rayleighDensity;
            PbrAtmosphereDensityProfile mieDensity;
            PbrAtmosphereDensityProfile absorptionDensity;
            f32                         sunAngularRadius;
            f32                         bottomRadius;
            f32                         topRadius;
            f32                         miePhaseFunctionG;
            f32                         muSMin;
            f32                         pad0;
            f32                         pad1;
            f32                         pad2;
            vec3                        solarIrradiance;
            vec3                        rayleighScattering;
            vec3                        mieScattering;
            vec3                        mieExtinction;
            vec3                        absorptionExtinction;
            vec3                        groundAlbedo;
        };
        using GPUBindId = Graphics::Rhi::RhiDescHandleLegacy;

        PbrAtmosphereParameter m_atmosphereParams;
        GPUBuffer              m_atmosphereParamsBuffer;

        GPUTexture             m_transmittance;
        GPUTexture             m_deltaIrradiance;
        GPUTexture             m_deltaRayleighScattering;
        GPUTexture             m_deltaMieScattering;
        GPUTexture             m_deltaScatteringDensity;
        GPUTexture             m_irradiance;
        GPUTexture             m_scattering;
        GPUTexture             m_deltaMultipleScattering;
        GPUTexture             m_optionalSingleMieScattering;

        Ref<GPUBindId>         m_transmittanceCombSamplerId;
        Ref<GPUBindId>         m_deltaIrradianceCombSamplerId;
        Ref<GPUBindId>         m_deltaRayleighScatteringCombSamplerId;
        Ref<GPUBindId>         m_deltaMieScatteringCombSamplerId;
        Ref<GPUBindId>         m_deltaScatteringDensityCombSamplerId;
        Ref<GPUBindId>         m_irradianceCombSamplerId;
        Ref<GPUBindId>         m_scatteringCombSamplerId;
        Ref<GPUBindId>         m_deltaMultipleScatteringCombSamplerId;
        Ref<GPUBindId>         m_optionalSingleMieScatteringCombSamplerId;

        Matrix4x4f             luminanceFromRad;
    };

    class IFRIT_APIDECL PbrAtmosphereRenderer : public RendererBase
    {
        using GPUCommandSubmission = Graphics::Rhi::RhiTaskSubmission;
        using RenderTargets        = Graphics::Rhi::RhiRenderTargets;
        using GPUShader            = Graphics::Rhi::RhiShader;
        using ComputePass          = Graphics::Rhi::RhiComputePass;

    private:
        ComputePass* m_transmittancePrecomputePass;
        ComputePass* m_irradiancePrecomputePass;
        ComputePass* m_singleScatteringPass;
        ComputePass* m_scatteringDensity;
        ComputePass* m_indirectIrradiancePass;
        ComputePass* m_multipleScatteringPass;

    protected:
        void       SetupTransmittancePrecomputePass();
        void       SetupIrradiancePrecomputePass();
        void       SetupSingleScatteringPass();
        void       SetupScatteringDensityPass();
        void       SetupIndirectIrradiancePass();
        void       SetupMultipleScatteringPass();

        void       PreparePerframeData(PerFrameData& perframeData);
        GPUShader* GetInternalShader(const char* name);

    public:
        PbrAtmosphereRenderer(IApplication* app) : RendererBase(app)
        {
            SetupTransmittancePrecomputePass();
            SetupIrradiancePrecomputePass();
            SetupSingleScatteringPass();
            SetupScatteringDensityPass();
            SetupIndirectIrradiancePass();
            SetupMultipleScatteringPass();
        }

        virtual Uref<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait) override
        {
            throw std::runtime_error("Not implemented");
        }

        virtual Uref<GPUCommandSubmission> RenderInternal(
            PerFrameData& perframe, const Vec<GPUCommandSubmission*>& cmdToWait);

        virtual PbrAtmosphereResourceDesc GetResourceDesc(PerFrameData& perframe);
    };

} // namespace Ifrit::Runtime