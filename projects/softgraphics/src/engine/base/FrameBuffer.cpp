
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

#include "ifrit/softgraphics/engine/base/FrameBuffer.h"
#include "ifrit/core/typing/Util.h"
using namespace Ifrit;

namespace Ifrit::Graphics::SoftGraphics
{
    IFRIT_APIDECL void FrameBuffer::SetColorAttachments(const std::vector<ImageF32*>& colorAttachment)
    {
        this->context->colorAttachment = colorAttachment;
        this->width                    = SizeCast<uint32_t>(colorAttachment[0]->GetWidth());
        this->height                   = SizeCast<uint32_t>(colorAttachment[0]->GetHeight());
    }
    IFRIT_APIDECL void FrameBuffer::SetDepthAttachment(ImageF32& depthAttachment)
    {
        this->context->depthAttachment = &depthAttachment;
    }

    IFRIT_APIDECL FrameBuffer::FrameBuffer() { this->context = new std::remove_pointer_t<decltype(this->context)>(); }
    IFRIT_APIDECL FrameBuffer::~FrameBuffer() { delete this->context; }

    /* DLL Compat */

    IFRIT_APIDECL void FrameBuffer::SetColorAttachmentsCompatible(ImageF32* const* colorAttachment, int nums)
    {
        this->context->colorAttachment = std::vector<ImageF32*>(nums);
        for (int i = 0; i < nums; i++)
        {
            this->context->colorAttachment[i] = colorAttachment[i];
        }
        this->width  = SizeCast<uint32_t>(colorAttachment[0]->GetWidth());
        this->height = SizeCast<uint32_t>(colorAttachment[0]->GetHeight());
    }
    IFRIT_APIDECL void FrameBuffer::SetDepthAttachmentCompatible(ImageF32* depthAttachment)
    {
        this->context->depthAttachment = depthAttachment;
    }
} // namespace Ifrit::Graphics::SoftGraphics