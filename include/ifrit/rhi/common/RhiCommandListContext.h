
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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
#include "RhiResource.h"
#include "RhiTransition.h"
#include "RhiApi.h"
#include "RhiDevice.h"

namespace Ifrit::RHI
{

    class RhiTaskSubmission
    {
    protected:
        virtual int _PolymorphismPlaceHolder() { return 0; }
    };

    struct RhiCommandListContextDesc
    {
        ERhiPipelineType mPipelineType = ERhiPipelineType::Graphics;
    };

    class IFRIT_RHI_API RhiCommandListContext : public RhiDeviceChild
    {
    public:
        virtual void CmdCopyBuffer(
            const RhiBuffer* srcBuffer, const RhiBuffer* dstBuffer, u32 size, u32 srcOffset, u32 dstOffset) const   = 0;
        virtual void CmdDispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const                           = 0;
        virtual void CmdSetViewports(const Vec<RhiViewport>& viewport) const                                        = 0;
        virtual void CmdSetScissors(const Vec<RhiScissor>& scissor) const                                           = 0;
        virtual void CmdDrawMeshTasks(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const                      = 0;
        virtual void CmdDrawMeshTasksIndirect(const RhiBuffer* buffer, u32 offset, u32 drawCount, u32 stride) const = 0;
        virtual void CmdDraw(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const          = 0;
        virtual void CmdDrawIndirect(const RhiBuffer* buffer, u32 offset) const                                     = 0;
        virtual void CmdDrawIndexed(
            u32 indexCount, u32 instanceCount, u32 firstIndex, i32 vertexOffset, u32 firstInstance) const           = 0;
        virtual void CmdDrawIndexedIndirect(const RhiBuffer* buffer, u32 offset) const                              = 0;
        virtual void CmdBufferClear(const RhiBuffer* buffer, u32 val) const                                         = 0;
        virtual void CmdAttachUniformRef(u32 setId, RhiBindlessDescriptorRef* ref) const                            = 0;
        virtual void CmdAttachVertexBufferView(const RhiVertexBufferView& view) const                               = 0;
        virtual void CmdAttachVertexBuffers(u32 firstSlot, const Vec<RhiBuffer*>& buffers) const                    = 0;
        virtual void CmdAttachIndexBuffer(const RhiBuffer* buffer) const                                            = 0;
        virtual void CmdDrawInstanced(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const = 0;
        virtual void CmdDispatchIndirect(const RhiBuffer* buffer, u32 offset) const                                 = 0;
        virtual void CmdSetPushConst(const void* data, u32 offset, u32 size) const                                  = 0;
        virtual void CmdClearUAVTexture(
            const RhiTexture* texture, RhiImageSubResource subResource, const RhiClearColorValue& clearValue) const = 0;
        virtual void CmdAddResourceBarrier(const Vec<RhiResourceBarrier>& barriers) const                           = 0;
        virtual void CmdGlobalMemoryBarrier() const                                                                 = 0;
        virtual void CmdCopyImage(const RhiTexture* src, RhiImageSubResource srcSub, const RhiTexture* dst,
            RhiImageSubResource dstSub) const                                                                       = 0;
        virtual void CmdCopyBufferToImage(
            const RhiBuffer* src, const RhiTexture* dst, RhiImageSubResource dstSub) const = 0;

        virtual void CmdSetCullMode(ERhiCullMode mode) const   = 0;
        virtual void CmdSetDepthFunc(ERhiDepthFunc func) const = 0;

        virtual void CmdBeginScope(const String& name) const = 0;
        virtual void CmdEndScope() const                     = 0;
    };
} // namespace Ifrit::RHI