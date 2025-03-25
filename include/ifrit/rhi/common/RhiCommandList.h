
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

namespace Ifrit::GraphicsBackend::Rhi {

class RhiTaskSubmission {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
};

class IFRIT_APIDECL RhiCommandList {
protected:
  RhiDevice *m_context;

protected:
  inline void _setTextureState(RhiTexture *texture, RhiResourceState state) const { texture->setState(state); }
  inline void _setBufferState(RhiBuffer *buffer, RhiResourceState state) const { buffer->setState(state); }

public:
  virtual void copyBuffer(const RhiBuffer *srcBuffer, const RhiBuffer *dstBuffer, u32 size, u32 srcOffset,
                          u32 dstOffset) const = 0;
  virtual void dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const = 0;
  virtual void setViewports(const Vec<RhiViewport> &viewport) const = 0;
  virtual void setScissors(const Vec<RhiScissor> &scissor) const = 0;
  virtual void drawMeshTasksIndirect(const RhiBuffer *buffer, u32 offset, u32 drawCount, u32 stride) const = 0;

  // Clear UAV storage buffer, considered as a transfer operation, typically
  // need a barrier for sync.
  virtual void bufferClear(const RhiBuffer *buffer, u32 val) const = 0;

  virtual void attachBindlessReferenceGraphics(RhiGraphicsPass *pass, u32 setId,
                                               RhiBindlessDescriptorRef *ref) const = 0;

  virtual void attachBindlessReferenceCompute(RhiComputePass *pass, u32 setId, RhiBindlessDescriptorRef *ref) const = 0;
  virtual void attachVertexBufferView(const RhiVertexBufferView &view) const = 0;
  virtual void attachVertexBuffers(u32 firstSlot, const Vec<RhiBuffer *> &buffers) const = 0;
  virtual void drawInstanced(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const = 0;
  virtual void dispatchIndirect(const RhiBuffer *buffer, u32 offset) const = 0;
  virtual void setPushConst(RhiComputePass *pass, u32 offset, u32 size, const void *data) const = 0;
  virtual void setPushConst(RhiGraphicsPass *pass, u32 offset, u32 size, const void *data) const = 0;

  virtual void clearUAVImageFloat(const RhiTexture *texture, RhiImageSubResource subResource,
                                  const Array<f32, 4> &val) const = 0;
  virtual void resourceBarrier(const Vec<RhiResourceBarrier> &barriers) const = 0;
  virtual void globalMemoryBarrier() const = 0;
  virtual void beginScope(const std::string &name) const = 0;
  virtual void endScope() const = 0;
  virtual void copyImage(const RhiTexture *src, RhiImageSubResource srcSub, const RhiTexture *dst,
                         RhiImageSubResource dstSub) const = 0;

  virtual void copyBufferToImage(const RhiBuffer *src, const RhiTexture *dst, RhiImageSubResource dstSub) const = 0;
  virtual void setCullMode(RhiCullMode mode) const = 0;
};

class IFRIT_APIDECL RhiQueue {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiQueue() = default;

  // Runs a command buffer, with CPU waiting
  // the GPU to finish
  virtual void runSyncCommand(Fn<void(const RhiCommandList *)> func) = 0;

  // Runs a command buffer, with CPU not
  // waiting the GPU to finish
  virtual Uref<RhiTaskSubmission> runAsyncCommand(Fn<void(const RhiCommandList *)> func,
                                                  const Vec<RhiTaskSubmission *> &waitOn,
                                                  const Vec<RhiTaskSubmission *> &toIssue) = 0;

  // Host sync
  virtual void hostWaitEvent(RhiTaskSubmission *event) = 0;
};
} // namespace Ifrit::GraphicsBackend::Rhi