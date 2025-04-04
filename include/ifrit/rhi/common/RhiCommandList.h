
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

#include <functional>

namespace Ifrit::Graphics::Rhi
{

    class RhiTaskSubmission
    {
    protected:
        virtual int _polymorphismPlaceHolder() { return 0; }
    };

    class IFRIT_APIDECL RhiCommandList
    {
    protected:
        RhiDevice* m_context;

    protected:
        inline void _setTextureState(RhiTexture* texture, RhiResourceState state) const { texture->SetState(state); }
        inline void _setBufferState(RhiBuffer* buffer, RhiResourceState state) const { buffer->SetState(state); }

    public:
        virtual void CopyBuffer(
            const RhiBuffer* srcBuffer, const RhiBuffer* dstBuffer, u32 size, u32 srcOffset, u32 dstOffset) const = 0;
        virtual void Dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const                            = 0;
        virtual void SetViewports(const Vec<RhiViewport>& viewport) const                                         = 0;
        virtual void SetScissors(const Vec<RhiScissor>& scissor) const                                            = 0;
        virtual void DrawMeshTasks(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const                       = 0;
        virtual void DrawMeshTasksIndirect(const RhiBuffer* buffer, u32 offset, u32 drawCount, u32 stride) const  = 0;
        virtual void DrawIndexed(
            u32 indexCount, u32 instanceCount, u32 firstIndex, int32_t vertexOffset, u32 firstInstance) const = 0;

        // Clear UAV storage buffer, considered as a transfer operation, typically
        // need a barrier for sync.
        virtual void BufferClear(const RhiBuffer* buffer, u32 val) const = 0;
        virtual void AttachBindlessRefGraphics(
            RhiGraphicsPass* pass, u32 setId, RhiBindlessDescriptorRef* ref) const                                  = 0;
        virtual void AttachBindlessRefCompute(RhiComputePass* pass, u32 setId, RhiBindlessDescriptorRef* ref) const = 0;
        virtual void AttachVertexBufferView(const RhiVertexBufferView& view) const                                  = 0;
        virtual void AttachVertexBuffers(u32 firstSlot, const Vec<RhiBuffer*>& buffers) const                       = 0;
        virtual void AttachIndexBuffer(const Rhi::RhiBuffer* buffer) const                                          = 0;
        virtual void DrawInstanced(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const    = 0;
        virtual void DispatchIndirect(const RhiBuffer* buffer, u32 offset) const                                    = 0;
        virtual void SetPushConst(RhiComputePass* pass, u32 offset, u32 size, const void* data) const               = 0;
        virtual void SetPushConst(RhiGraphicsPass* pass, u32 offset, u32 size, const void* data) const              = 0;
        virtual void ClearUAVTexFloat(
            const RhiTexture* texture, RhiImageSubResource subResource, const Array<f32, 4>& val) const = 0;
        virtual void AddResourceBarrier(const Vec<RhiResourceBarrier>& barriers) const                  = 0;
        virtual void GlobalMemoryBarrier() const                                                        = 0;
        virtual void BeginScope(const String& name) const                                               = 0;
        virtual void EndScope() const                                                                   = 0;
        virtual void CopyImage(const RhiTexture* src, RhiImageSubResource srcSub, const RhiTexture* dst,
            RhiImageSubResource dstSub) const                                                           = 0;
        virtual void CopyBufferToImage(
            const RhiBuffer* src, const RhiTexture* dst, RhiImageSubResource dstSub) const = 0;
        virtual void SetCullMode(RhiCullMode mode) const                                   = 0;
    };

    class IFRIT_APIDECL RhiQueue
    {
    protected:
        RhiDevice* m_context;

    public:
        virtual ~RhiQueue() = default;

        // Runs a command buffer, with CPU waiting the GPU to finish
        virtual void                    RunSyncCommand(Fn<void(const RhiCommandList*)> func) = 0;

        // Runs a command buffer, with CPU not waiting the GPU to finish
        virtual Uref<RhiTaskSubmission> RunAsyncCommand(Fn<void(const RhiCommandList*)> func,
            const Vec<RhiTaskSubmission*>& waitOn, const Vec<RhiTaskSubmission*>& toIssue) = 0;

        // Host sync
        virtual void                    HostWaitEvent(RhiTaskSubmission* event) = 0;
    };
} // namespace Ifrit::Graphics::Rhi