
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
#include "ifrit/vkrhi/engine/vkrenderer/EngineContext.h"
#include "ifrit/vkrhi/engine/vkrenderer/MemoryResource.h"

namespace Ifrit::RHI::VulkanAdapter
{
    class IFRIT_APIDECL StagedSingleBuffer : public RHI::RhiStagedSingleBuffer
    {
    protected:
        RHI::RhiBufferRef m_bufferUnique;
        SingleBuffer*     m_buffer;
        RHI::RhiBufferRef m_stagingBuffer;
        EngineContext*    m_context;

    public:
        StagedSingleBuffer(EngineContext* ctx, SingleBuffer* buffer);
        StagedSingleBuffer(EngineContext* ctx, const BufferCreateInfo& ci);
        StagedSingleBuffer(const StagedSingleBuffer& p)            = delete;
        StagedSingleBuffer& operator=(const StagedSingleBuffer& p) = delete;

        virtual ~StagedSingleBuffer() {}
        void CmdCopyToDevice(const RHI::RhiCommandList* cmd, const void* data, u32 size, u32 localOffset) override;
    };

    class IFRIT_APIDECL StagedSingleImage
    {
    protected:
        RHI::RhiTextureRef m_imageUnique;
        SingleDeviceImage* m_image;
        RHI::RhiBufferRef  m_stagingBuffer;
        EngineContext*     m_context;

    public:
        StagedSingleImage(EngineContext* ctx, SingleDeviceImage* image);
        StagedSingleImage(EngineContext* ctx, const ImageCreateInfo& ci);

        StagedSingleImage(const StagedSingleImage& p)            = delete;
        StagedSingleImage& operator=(const StagedSingleImage& p) = delete;

        virtual ~StagedSingleImage() {}
        void CmdCopyToDevice(CommandBuffer* cmd, const void* data, VkImageLayout srcLayout, VkImageLayout dstlayout,
            VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);
    };
} // namespace Ifrit::RHI::VulkanAdapter