
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
#include "ifrit/core/algo/Hash.h"
#include "ifrit/runtime/renderer/PostprocessPass.h"
#include <unordered_map>

namespace Ifrit::Runtime::PostprocessPassCollection
{

    struct PostFxFFTConv2dSubpasses;

    struct PostFxFFTConv2dResourceCollection
    {
        using GPUBindId  = Graphics::Rhi::RhiDescHandleLegacy;
        using GPUTexture = Graphics::Rhi::RhiTextureRef;
        using GPUShader  = Graphics::Rhi::RhiShader;
        using GPUSampler = Graphics::Rhi::RhiSamplerRef;

        GPUTexture     m_tex1;
        Ref<GPUBindId> m_tex1IdSamp;
        GPUTexture     m_tex2;
        GPUTexture     m_texTemp;

        GPUTexture     m_texGaussian;
        GPUSampler     m_texGaussianSampler;
        Ref<GPUBindId> m_texGaussianSampId;

        PostFxFFTConv2dResourceCollection()                                         = default;
        PostFxFFTConv2dResourceCollection(const PostFxFFTConv2dResourceCollection&) = default;
    };

    class IFRIT_APIDECL PostFxFFTConv2d : public PostprocessPass
    {
        using GPUBindId     = Graphics::Rhi::RhiDescHandleLegacy;
        using RenderTargets = Graphics::Rhi::RhiRenderTargets;
        using GPUTexture    = Graphics::Rhi::RhiTexture;
        using ComputePass   = Graphics::Rhi::RhiComputePass;

        // I know this is UGLY
        CustomHashMap<Pair<u32, u32>, PostFxFFTConv2dResourceCollection, PairwiseHash<u32, u32>> m_resMap;

        ComputePass* m_upsamplePipeline = nullptr;
        ComputePass* m_gaussianPipeline = nullptr;

    public:
        PostFxFFTConv2d(IApplication* app);
        ~PostFxFFTConv2d();
        void RenderPostFx(const GPUCmdBuffer* cmd, GPUBindId* srcSampId, u32 dstUAVImg, GPUBindId* kernelSampId,
            u32 srcWidth, u32 srcHeight, u32 kernelWidth, u32 kernelHeight, u32 srcDownscale);
    };

} // namespace Ifrit::Runtime::PostprocessPassCollection