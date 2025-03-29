
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
    struct IFRIT_APIDECL RhiRenderTargetsFormat
    {
        RhiImageFormat      m_depthFormat;
        Vec<RhiImageFormat> m_colorFormats;
    };

    class IFRIT_APIDECL RhiRenderTargets
    {
    public:
        virtual void                       SetColorAttachments(const Vec<RhiColorAttachment*>& attachments) = 0;
        virtual void                       SetDepthStencilAttachment(RhiDepthStencilAttachment* attachment) = 0;
        virtual void                       BeginRendering(const RhiCommandList* commandBuffer) const        = 0;
        virtual void                       EndRendering(const RhiCommandList* commandBuffer) const          = 0;
        virtual void                       SetRenderArea(RhiScissor area)                                   = 0;
        virtual RhiRenderTargetsFormat     GetFormat() const                                                = 0;
        virtual RhiScissor                 GetRenderArea() const                                            = 0;
        virtual RhiDepthStencilAttachment* GetDepthStencilAttachment() const                                = 0;
        virtual RhiColorAttachment*        GetColorAttachment(u32 index) const                              = 0;
    };

    class IFRIT_APIDECL RhiColorAttachment
    {
    public:
        virtual RhiTexture* GetRenderTarget() const                          = 0;
        virtual void        SetBlendInfo(const RhiAttachmentBlendInfo& info) = 0;
    };

    class IFRIT_APIDECL RhiDepthStencilAttachment
    {
    public:
        virtual RhiTexture* GetTexture() const = 0;
    };

    class IFRIT_APIDECL RhiVertexBufferView
    {
    public:
        virtual void AddBinding(Vec<u32> location, Vec<RhiImageFormat> format, Vec<u32> offset, u32 stride,
            RhiVertexInputRate inputRate = RhiVertexInputRate::Vertex) = 0;
    };
} // namespace Ifrit::Graphics::Rhi