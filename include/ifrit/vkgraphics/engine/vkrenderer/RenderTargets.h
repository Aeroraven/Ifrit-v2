
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
#include "ifrit/core/typing/Util.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"

namespace Ifrit::Graphics::VulkanGraphics
{
    class IFRIT_APIDECL ColorAttachment : public Rhi::RhiColorAttachment
    {
    private:
        SingleDeviceImage*          m_renderTarget;
        Rhi::RhiClearValue          m_clearValue;
        Rhi::RhiRenderTargetLoadOp  m_loadOp;
        u32                         m_targetMip      = ~0u;
        u32                         m_targetArrLayer = ~0u;
        Rhi::RhiAttachmentBlendInfo m_blendInfo;

    public:
        ColorAttachment(Rhi::RhiTexture* renderTarget, Rhi::RhiClearValue clearValue, Rhi::RhiRenderTargetLoadOp loadOp,
            u32 mip, u32 arrLayer)
            : m_renderTarget(Ifrit::CheckedCast<SingleDeviceImage>(renderTarget))
            , m_clearValue(clearValue)
            , m_loadOp(loadOp)
            , m_targetMip(mip)
            , m_targetArrLayer(arrLayer)
        {
        }

        inline SingleDeviceImage*         GetRenderTargetInternal() const { return m_renderTarget; }
        inline Rhi::RhiTexture*           GetRenderTarget() const override { return m_renderTarget; }
        inline Rhi::RhiClearValue         GetClearValue() const { return m_clearValue; }
        inline Rhi::RhiRenderTargetLoadOp GetLoadOp() const { return m_loadOp; }
        inline u32                        GetTargetMip() const { return m_targetMip; }
        inline u32                        GetTargetArrLayer() const { return m_targetArrLayer; }

        inline void SetBlendInfo(const Rhi::RhiAttachmentBlendInfo& info) override { m_blendInfo = info; }
        inline Rhi::RhiAttachmentBlendInfo GetBlendInfo() const { return m_blendInfo; }
    };

    class IFRIT_APIDECL DepthStencilAttachment : public Rhi::RhiDepthStencilAttachment
    {
    private:
        SingleDeviceImage*         m_renderTarget;
        Rhi::RhiClearValue         m_clearValue;
        Rhi::RhiRenderTargetLoadOp m_loadOp;

    public:
        DepthStencilAttachment(
            Rhi::RhiTexture* renderTarget, Rhi::RhiClearValue clearValue, Rhi::RhiRenderTargetLoadOp loadOp)
            : m_renderTarget(Ifrit::CheckedCast<SingleDeviceImage>(renderTarget))
            , m_clearValue(clearValue)
            , m_loadOp(loadOp)
        {
        }

        inline SingleDeviceImage*         GetRenderTargetInternal() const { return m_renderTarget; }
        inline Rhi::RhiTexture*           GetRenderTarget() const { return m_renderTarget; }
        inline Rhi::RhiTexture*           GetTexture() const override { return m_renderTarget; }
        inline Rhi::RhiClearValue         GetClearValue() const { return m_clearValue; }
        inline Rhi::RhiRenderTargetLoadOp GetLoadOp() const { return m_loadOp; }
    };

    class IFRIT_APIDECL RenderTargets : public Rhi::RhiRenderTargets
    {
    private:
        Vec<ColorAttachment*>   m_colorAttachments;
        DepthStencilAttachment* m_depthStencilAttachment = nullptr;
        EngineContext*          m_context;
        Rhi::RhiScissor         m_renderArea;

    public:
        RenderTargets(EngineContext* context) : m_context(context) {}
        ~RenderTargets() = default;

        inline void                 SetRenderArea(Rhi::RhiScissor area) override { m_renderArea = area; }
        void                        SetColorAttachments(const Vec<Rhi::RhiColorAttachment*>& attachments) override;
        void                        SetDepthStencilAttachment(Rhi::RhiDepthStencilAttachment* attachment) override;
        void                        BeginRendering(const Rhi::RhiCommandList* commandBuffer) const override;
        void                        EndRendering(const Rhi::RhiCommandList* commandBuffer) const override;
        Rhi::RhiRenderTargetsFormat GetFormat() const override;
        virtual Rhi::RhiScissor     GetRenderArea() const override;
        inline Rhi::RhiDepthStencilAttachment* GetDepthStencilAttachment() const override
        {
            return m_depthStencilAttachment;
        }
        inline Rhi::RhiColorAttachment* GetColorAttachment(u32 index) const { return m_colorAttachments[index]; }
    };
} // namespace Ifrit::Graphics::VulkanGraphics