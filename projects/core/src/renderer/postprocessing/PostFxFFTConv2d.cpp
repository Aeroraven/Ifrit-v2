
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
#include "ifrit/core/renderer/postprocessing/PostFxFFTConv2d.h"
#include "ifrit.shader/Postprocess/FFTConv2d.Shared.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/math/fastutil/FastUtil.h"
#include "ifrit/core/renderer/postprocessing/PostFxStockhamDFT2.h"

namespace Ifrit::Core::PostprocessPassCollection
{

    IFRIT_APIDECL PostFxFFTConv2d::PostFxFFTConv2d(IApplication* app)
        : PostprocessPass(app, { "FFTConv2d.comp.glsl", 17, 1, true })
    {
        using namespace Ifrit::Graphics::Rhi;
        auto rhi           = app->GetRhi();
        m_upsamplePipeline = rhi->CreateComputePass();

        auto shader = CreateShaderFromFile("FFTConv2d.Upsample.comp.glsl", "main", RhiShaderStage::Compute);
        m_upsamplePipeline->SetComputeShader(shader);
        m_upsamplePipeline->SetPushConstSize(17 * sizeof(u32));
        m_upsamplePipeline->SetNumBindlessDescriptorSets(0);

        auto gshader       = CreateShaderFromFile("GaussianKernelGenerate.comp.glsl", "main", RhiShaderStage::Compute);
        m_gaussianPipeline = rhi->CreateComputePass();
        m_gaussianPipeline->SetComputeShader(gshader);
        m_gaussianPipeline->SetPushConstSize(4 * sizeof(u32));
        m_gaussianPipeline->SetNumBindlessDescriptorSets(0);
    }

    IFRIT_APIDECL      PostFxFFTConv2d::~PostFxFFTConv2d() {}

    IFRIT_APIDECL void PostFxFFTConv2d::RenderPostFx(const GPUCmdBuffer* cmd, GPUBindId* srcSampId, u32 dstUAVImg,
        GPUBindId* kernelSampId, u32 srcWidth, u32 srcHeight, u32 kernelWidth,
        u32 kernelHeight, u32 srcDownscale)
    {
        // todo

        auto imageX = srcWidth / srcDownscale, imageY = srcHeight / srcDownscale;
        auto imagePaddingX = kernelWidth / 2, imagePaddingY = kernelHeight / 2;

        auto imagePaddedX = imageX + imagePaddingX * 2, imagePaddedY = imageY + imagePaddingY * 2;

        auto kenelPaddingXLo = imagePaddedX / 2 - imageX / 2;
        auto kenelPaddingYLo = imagePaddedY / 2 - imageY / 2;
        auto kenelPaddingXHi = imagePaddedX - imageX - kenelPaddingXLo;
        auto kenelPaddingYHi = imagePaddedY - imageY - kenelPaddingYLo;

        using Ifrit::Math::CountLeadingZero;
        using Ifrit::Math::IntegerLog2;
        auto logP2Width  = 32 - CountLeadingZero(imagePaddedX - 1);
        auto logP2Height = 32 - CountLeadingZero(imagePaddedY - 1);

        logP2Width  = std::max(logP2Width, logP2Height);
        logP2Height = logP2Width;

        auto p2Width  = 1 << logP2Width;
        auto p2Height = 1 << logP2Height;

        if (p2Width > FFTConv2DConfig::cMaxSupportedTextureSize || p2Height > FFTConv2DConfig::cMaxSupportedTextureSize)
        {
            iInfo("Padded image size is: {}x{}", imagePaddedX, imagePaddedY);
            iError("FFTConv2d: Image size too large, max {}x{}. Got {}x{}", FFTConv2DConfig::cMaxSupportedTextureSize,
                FFTConv2DConfig::cMaxSupportedTextureSize, p2Width, p2Height);
            return;
        }
        bool firstTime = false;
        if (m_resMap.count({ p2Width, p2Height }) == 0)
        {
            firstTime = true;
            using namespace Ifrit::Graphics::Rhi;
            auto res  = std::make_unique<PostFxFFTConv2dResourceCollection>();
            auto rhi  = m_app->GetRhi();
            auto tex1 = rhi->CreateTexture2D(
                "PostFx_Conv_Tex1", p2Width * 2, p2Height, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT, true);
            auto tex2 = rhi->CreateTexture2D(
                "PostFx_Conv_Tex2", p2Width * 2, p2Height, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT, true);
            auto texTemp = rhi->CreateTexture2D(
                "PostFx_Conv_TexTemp", p2Width * 2, p2Height, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT, true);

            auto texSampler  = rhi->CreateTrivialSampler();
            auto texGaussian = rhi->CreateTexture2D(
                "PostFx_Conv_TexGaussian", kernelWidth, kernelHeight, RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT, true);
            auto texGaussianSampId = rhi->RegisterCombinedImageSampler(texGaussian.get(), texSampler.get());

            res->m_tex1               = tex1;
            res->m_tex1IdSamp         = rhi->RegisterCombinedImageSampler(tex1.get(), texSampler.get());
            res->m_tex2               = tex2;
            res->m_texTemp            = texTemp;
            res->m_texGaussian        = texGaussian;
            res->m_texGaussianSampler = texSampler;
            res->m_texGaussianSampId  = texGaussianSampId;

            m_resMap[{ p2Width, p2Height }] = *res;
        }

        struct PushConst
        {
            u32 srcDownScale;
            u32 kernDownScale;
            u32 srcRtW;
            u32 srcRtH;
            u32 kernRtW;
            u32 kernRtH;
            u32 srcImage;
            u32 srcIntermImage;
            u32 kernImage;
            u32 kernIntermImage;
            u32 dstImage;
            u32 tempImage;
            u32 fftTexSizeWLog;
            u32 fftTexSizeHLog;
            u32 fftStep;
            u32 bloomMix;
            u32 srcIntermImageSamp;
        } pc;

        pc.srcDownScale       = srcDownscale;
        pc.kernDownScale      = 1;
        pc.srcRtW             = srcWidth;
        pc.srcRtH             = srcHeight;
        pc.kernRtW            = kernelWidth;
        pc.kernRtH            = kernelHeight;
        pc.srcImage           = srcSampId->GetActiveId();
        pc.srcIntermImage     = m_resMap[{ p2Width, p2Height }].m_tex1->GetDescId();
        pc.srcIntermImageSamp = m_resMap[{ p2Width, p2Height }].m_tex1IdSamp->GetActiveId();
        pc.kernImage          = 0; // kernelSampId->GetActiveId();
        pc.kernIntermImage    = m_resMap[{ p2Width, p2Height }].m_tex2->GetDescId();
        pc.dstImage           = dstUAVImg;
        pc.tempImage          = m_resMap[{ p2Width, p2Height }].m_texTemp->GetDescId();
        pc.fftTexSizeWLog     = logP2Width;
        pc.fftTexSizeHLog     = logP2Height;
        pc.bloomMix           = 1;

        if (kernelSampId != nullptr)
        {
            pc.kernImage = kernelSampId->GetActiveId();
        }
        else
        {
            pc.kernImage = m_resMap[{ p2Width, p2Height }].m_texGaussianSampId->GetActiveId();

            struct PushConstBlur
            {
                u32   blurKernW;
                u32   blurKernH;
                u32   srcImgId;
                float sigma = 10.0f;
            } pcb;
            pcb.blurKernW = kernelWidth;
            pcb.blurKernH = kernelHeight;
            pcb.srcImgId  = m_resMap[{ p2Width, p2Height }].m_texGaussian->GetDescId();
            if (firstTime)
            {
                cmd->BeginScope("Postprocess: FFTConv2D, GaussianBlur");
                m_gaussianPipeline->SetRecordFunction([&](const Graphics::Rhi::RhiRenderPassContext* ctx) {
                    cmd->SetPushConst(m_gaussianPipeline, 0, 4 * sizeof(u32), &pcb);
                    ctx->m_cmd->Dispatch(Ifrit::Math::DivRoundUp(p2Width, 8), Ifrit::Math::DivRoundUp(p2Height, 8), 1);
                });
                m_gaussianPipeline->Run(cmd, 0);
                cmd->GlobalMemoryBarrier();
                cmd->EndScope();
            }
        }

        if (logP2Height != logP2Width)
        {
            iError("FFTConv2d: Image size must be square");
            return;
        }
        auto wgX = Ifrit::Math::DivRoundUp(p2Width / 2, FFTConv2DConfig::cThreadGroupSizeX);
        auto wgY = p2Height;
        cmd->BeginScope("Postprocess: FFTConv2D");
        const char* scopeNames[7] = {
            "Postprocess: FFTConv2D, SrcFFT-X",
            "Postprocess: FFTConv2D, SrcFFT-Y",
            "Postprocess: FFTConv2D, KernFFT-X",
            "Postprocess: FFTConv2D, KernFFT-Y",
            "Postprocess: FFTConv2D, Multiply",
            "Postprocess: FFTConv2D, IFFT-Y",
            "Postprocess: FFTConv2D, IFFT-X",
        };
        for (u32 i = 0u; i < 7u; i++)
        {
            if (!firstTime)
            {
                if (i == 2u || i == 3u)
                    continue;
            }
            cmd->BeginScope(scopeNames[i]);
            m_computePipeline->SetRecordFunction([&](const Graphics::Rhi::RhiRenderPassContext* ctx) {
                pc.fftStep = i;
                cmd->SetPushConst(m_computePipeline, 0, 16 * sizeof(u32), &pc);
                ctx->m_cmd->Dispatch(wgX, wgY, 1);
            });
            m_computePipeline->Run(cmd, 0);
            cmd->GlobalMemoryBarrier();
            cmd->EndScope();
        }

        cmd->BeginScope("Postprocess: FFTConv2D, Upsample");
        auto dwgX = Ifrit::Math::DivRoundUp(srcWidth, 8);
        auto dwgY = Ifrit::Math::DivRoundUp(srcHeight, 8);
        m_upsamplePipeline->SetRecordFunction([&](const Graphics::Rhi::RhiRenderPassContext* ctx) {
            pc.tempImage = m_resMap[{ p2Width, p2Height }].m_texTemp->GetDescId();
            cmd->SetPushConst(m_upsamplePipeline, 0, 17 * sizeof(u32), &pc);
            ctx->m_cmd->Dispatch(dwgX, dwgY, 1);
        });
        m_upsamplePipeline->Run(cmd, 0);
        cmd->GlobalMemoryBarrier();
        cmd->EndScope();
        cmd->EndScope();
    }
} // namespace Ifrit::Core::PostprocessPassCollection