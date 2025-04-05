
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

#include "RhiBaseTypes.h"

namespace Ifrit::Graphics::Rhi
{
    struct IFRIT_APIDECL RhiRenderPassContext
    {
        const RhiCommandList* m_cmd;
        u32                   m_frame;
    };

    class IFRIT_APIDECL RhiGeneralPassBase
    {
    };

    class IFRIT_APIDECL RhiComputePass : public RhiGeneralPassBase
    {

    public:
        virtual ~RhiComputePass()                                                                          = default;
        virtual void SetComputeShader(RhiShader* shader)                                                   = 0;
        virtual void SetShaderBindingLayout(const Vec<RhiDescriptorType>& layout)                          = 0;
        virtual void AddShaderStorageBuffer(RhiBuffer* buffer, u32 position, RhiResourceAccessType access) = 0;
        virtual void AddUniformBuffer(RhiMultiBuffer* buffer, u32 position)                                = 0;
        virtual void SetExecutionFunction(Fn<void(RhiRenderPassContext*)> func)                            = 0;
        virtual void SetRecordFunction(Fn<void(RhiRenderPassContext*)> func)                               = 0;

        virtual void Run(const RhiCommandList* cmd, u32 frameId) = 0;
        virtual void SetNumBindlessDescriptorSets(u32 num)       = 0;
        virtual void SetPushConstSize(u32 size)                  = 0;
    };

    class IFRIT_APIDECL RhiGraphicsPass : public RhiGeneralPassBase
    {

    public:
        virtual ~RhiGraphicsPass()                                         = default;
        virtual void SetTaskShader(RhiShader* shader)                      = 0;
        virtual void SetMeshShader(RhiShader* shader)                      = 0;
        virtual void SetVertexShader(RhiShader* shader)                    = 0;
        virtual void SetPixelShader(RhiShader* shader)                     = 0;
        virtual void SetRasterizerTopology(RhiRasterizerTopology topology) = 0;
        virtual void SetRenderArea(u32 x, u32 y, u32 width, u32 height)    = 0;
        virtual void SetDepthWrite(bool write)                             = 0;
        virtual void SetDepthTestEnable(bool enable)                       = 0;
        virtual void SetDepthCompareOp(RhiCompareOp compareOp)             = 0;
        virtual void SetMsaaSamples(u32 samples)                           = 0;

        virtual void SetRenderTargetFormat(const RhiRenderTargetsFormat& format)                           = 0;
        virtual void SetShaderBindingLayout(const Vec<RhiDescriptorType>& layout)                          = 0;
        virtual void AddShaderStorageBuffer(RhiBuffer* buffer, u32 position, RhiResourceAccessType access) = 0;
        virtual void AddUniformBuffer(RhiMultiBuffer* buffer, u32 position)                                = 0;
        virtual void SetExecutionFunction(Fn<void(RhiRenderPassContext*)> func)                            = 0;
        virtual void SetRecordFunction(Fn<void(RhiRenderPassContext*)> func)                               = 0;
        virtual void SetRecordFunctionPostRenderPass(Fn<void(RhiRenderPassContext*)> func)                 = 0;

        virtual void Run(const RhiCommandList* cmd, RhiRenderTargets* renderTargets, u32 frameId) = 0;
        virtual void SetNumBindlessDescriptorSets(u32 num)                                        = 0;
        virtual void SetPushConstSize(u32 size)                                                   = 0;
    };

    class IFRIT_APIDECL RhiRTPipeline
    {
    public:
        virtual void _polymorphismPlaceHolder() {}
    };

    class IFRIT_APIDECL RhiRTPass : public RhiGeneralPassBase
    {
    public:
        virtual void _polymorphismPlaceHolder() {}
    };

} // namespace Ifrit::Graphics::Rhi