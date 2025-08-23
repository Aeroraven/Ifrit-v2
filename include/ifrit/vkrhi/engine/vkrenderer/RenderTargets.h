
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
#include "ifrit/vkrhi/common/Pch.h"
#include "ifrit/vkrhi/engine/vkrenderer/Command.h"
#include "ifrit/vkrhi/engine/vkrenderer/MemoryResource.h"

namespace Ifrit::RHI::VulkanAdapter
{
    class IFRIT_APIDECL ColorAttachment : public RHI::RhiColorAttachment
    {
    private:
        SingleDeviceImage*          m_renderTarget;
        RHI::RhiClearValue2         m_clearValue;
        RHI::RhiRenderTargetLoadOp  m_loadOp;
        u32                         m_targetMip      = ~0u;
        u32                         m_targetArrLayer = ~0u;
        RHI::RhiAttachmentBlendInfo m_blendInfo;

    public:
        ColorAttachment(RHI::RhiTexture* renderTarget, RHI::RhiClearValue2 clearValue,
            RHI::RhiRenderTargetLoadOp loadOp, u32 mip, u32 arrLayer)
            : m_renderTarget(Ifrit::CheckedCast<SingleDeviceImage>(renderTarget))
            , m_clearValue(clearValue)
            , m_loadOp(loadOp)
            , m_targetMip(mip)
            , m_targetArrLayer(arrLayer)
        {
        }

        inline SingleDeviceImage*         GetRenderTargetInternal() const { return m_renderTarget; }
        inline RHI::RhiTexture*           GetRenderTarget() const override { return m_renderTarget; }
        inline RHI::RhiClearValue2        GetClearValue() const { return m_clearValue; }
        inline RHI::RhiRenderTargetLoadOp GetLoadOp() const { return m_loadOp; }
        inline u32                        GetTargetMip() const { return m_targetMip; }
        inline u32                        GetTargetArrLayer() const { return m_targetArrLayer; }

        inline void SetBlendInfo(const RHI::RhiAttachmentBlendInfo& info) override { m_blendInfo = info; }
        inline RHI::RhiAttachmentBlendInfo GetBlendInfo() const { return m_blendInfo; }
    };

    class IFRIT_APIDECL DepthStencilAttachment : public RHI::RhiDepthStencilAttachment
    {
    private:
        SingleDeviceImage*         m_renderTarget;
        RHI::RhiClearValue2        m_clearValue;
        RHI::RhiRenderTargetLoadOp m_loadOp;

    public:
        DepthStencilAttachment(
            RHI::RhiTexture* renderTarget, RHI::RhiClearValue2 clearValue, RHI::RhiRenderTargetLoadOp loadOp)
            : m_renderTarget(Ifrit::CheckedCast<SingleDeviceImage>(renderTarget))
            , m_clearValue(clearValue)
            , m_loadOp(loadOp)
        {
        }

        inline SingleDeviceImage*         GetRenderTargetInternal() const { return m_renderTarget; }
        inline RHI::RhiTexture*           GetRenderTarget() const { return m_renderTarget; }
        inline RHI::RhiTexture*           GetTexture() const override { return m_renderTarget; }
        inline RHI::RhiClearValue2        GetClearValue() const { return m_clearValue; }
        inline RHI::RhiRenderTargetLoadOp GetLoadOp() const { return m_loadOp; }
    };

    class IFRIT_APIDECL RenderTargets : public RHI::RhiRenderTargets
    {
    private:
        Vec<ColorAttachment*>   m_colorAttachments;
        DepthStencilAttachment* m_depthStencilAttachment = nullptr;
        EngineContext*          m_context;
        RHI::RhiScissor         m_renderArea;

    public:
        RenderTargets(EngineContext* context) : m_context(context) {}
        ~RenderTargets() = default;

        inline void                 SetRenderArea(RHI::RhiScissor area) override { m_renderArea = area; }
        void                        SetColorAttachments(const Vec<RHI::RhiColorAttachment*>& attachments) override;
        void                        SetDepthStencilAttachment(RHI::RhiDepthStencilAttachment* attachment) override;
        void                        BeginRendering(const RHI::RhiCommandListContext* commandBuffer) const override;
        void                        EndRendering(const RHI::RhiCommandListContext* commandBuffer) const override;
        RHI::RhiRenderTargetsFormat GetFormat() const override;
        virtual RHI::RhiScissor     GetRenderArea() const override;
        inline RHI::RhiDepthStencilAttachment* GetDepthStencilAttachment() const override
        {
            return m_depthStencilAttachment;
        }
        inline RHI::RhiColorAttachment* GetColorAttachment(u32 index) const { return m_colorAttachments[index]; }
    };
} // namespace Ifrit::RHI::VulkanAdapter