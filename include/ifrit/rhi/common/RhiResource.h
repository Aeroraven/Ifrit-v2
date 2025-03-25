
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
#include "ifrit/common/logging/Logging.h"
#include <queue>

namespace Ifrit::GraphicsBackend::Rhi {

// UPD 250325: Resource removal algo before destroys the resource that still in use on device side
// referencing Unreal's resource state management, a delete queue should be maintained

class IFRIT_APIDECL IRhiDeviceResourceDeleteQueue {
public:
  virtual void addResourceToDeleteQueue(RhiDeviceResource *resource) = 0;
  virtual i32 processDeleteQueue() = 0;
};

class IFRIT_APIDECL RhiDeviceResource {
private:
  Atomic<u32> m_refCount = 0;
  IRhiDeviceResourceDeleteQueue *m_deleteQueue;
  u32 m_bindlessId;
  bool m_isUnmanaged = false; // unmanaged resources are EXTERNAL resources
  std::string m_debugName;

public:
  explicit RhiDeviceResource(nullptr_t v) : m_deleteQueue(nullptr), m_isUnmanaged(true) {}
  RhiDeviceResource(IRhiDeviceResourceDeleteQueue *deleteQueue) : m_deleteQueue(deleteQueue) {}
  virtual ~RhiDeviceResource() {}

  inline virtual void addRef() { m_refCount.fetch_add(1); }
  inline virtual void release() {
    if (m_refCount.fetch_sub(1) == 1) {
      if (!m_isUnmanaged) {
        markForDelete();
      }
    }
  }
  inline virtual void markForDelete() { m_deleteQueue->addResourceToDeleteQueue(this); }
  inline virtual void setResId(u32 id) { m_bindlessId = id; }
  inline virtual u32 getResId() const { return m_bindlessId; }
  inline virtual void setDebugName(const std::string &name) { m_debugName = name; }
  inline virtual const std::string &getDebugName() const { return m_debugName; }
};

class IFRIT_APIDECL RhiBuffer : public RhiDeviceResource {
protected:
  RhiDevice *m_context;
  RhiResourceState m_state = RhiResourceState::Undefined;

private:
  inline void setState(RhiResourceState state) { m_state = state; }

public:
  RhiBuffer(IRhiDeviceResourceDeleteQueue *deleteQueue) : RhiDeviceResource(deleteQueue) {}
  virtual ~RhiBuffer() = default;
  virtual void map() = 0;
  virtual void unmap() = 0;
  virtual void flush() = 0;
  virtual void readBuffer(void *data, u32 size, u32 offset) = 0;
  virtual void writeBuffer(const void *data, u32 size, u32 offset) = 0;
  virtual inline RhiResourceState getState() const { return m_state; }

  virtual RhiDeviceAddr getDeviceAddress() const = 0;

  friend class RhiCommandList;
};

class IFRIT_APIDECL RhiMultiBuffer {
protected:
  RhiDevice *m_context;
  IRhiDeviceResourceDeleteQueue *m_deleteQueue;

public:
  RhiMultiBuffer(IRhiDeviceResourceDeleteQueue *deleteQueue) : m_deleteQueue(deleteQueue) {}
  virtual RhiBuffer *getActiveBuffer() = 0;
  virtual RhiBuffer *getActiveBufferRelative(u32 deltaFrame) = 0;
  virtual ~RhiMultiBuffer() = default;
};

class IFRIT_APIDECL RhiStagedSingleBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiStagedSingleBuffer() = default;
  virtual void cmdCopyToDevice(const RhiCommandList *cmd, const void *data, u32 size, u32 localOffset) = 0;
};

class RhiStagedMultiBuffer {};

class IFRIT_APIDECL RhiTexture : public RhiDeviceResource {
protected:
  RhiDevice *m_context;
  RhiResourceState m_state = RhiResourceState::Undefined;
  bool m_rhiSwapchainImage = false;

private:
  inline void setState(RhiResourceState state) { m_state = state; }

public:
  explicit RhiTexture(nullptr_t v) : RhiDeviceResource(nullptr) {}
  RhiTexture(IRhiDeviceResourceDeleteQueue *deleteQueue) : RhiDeviceResource(deleteQueue) {}
  virtual ~RhiTexture() = default;
  virtual u32 getHeight() const = 0;
  virtual u32 getWidth() const = 0;
  virtual bool isDepthTexture() const = 0;
  virtual inline RhiResourceState getState() const { return m_state; }
  virtual void *getNativeHandle() const = 0;

  friend class RhiCommandList;
};

class IFRIT_APIDECL RhiSampler : public RhiDeviceResource {
protected:
  RhiSampler(IRhiDeviceResourceDeleteQueue *deleteQueue) : RhiDeviceResource(deleteQueue) {}
  virtual int _polymorphismPlaceHolder() { return 0; }
};

struct IFRIT_APIDECL RhiRTGeometryReference {
  RhiDeviceAddr m_vertex;
  RhiDeviceAddr m_index;
  RhiDeviceAddr m_transform;
  u32 m_numVertices;
  u32 m_numIndices;
  u32 m_vertexComponents = 3;
  u32 m_vertexStride = 12;
};

class IFRIT_APIDECL RhiRTInstance {
public:
  virtual RhiDeviceAddr getDeviceAddress() const = 0;
};

class IFRIT_APIDECL RhiRTScene {
public:
  virtual RhiDeviceAddr getDeviceAddress() const = 0;
};

class IFRIT_APIDECL RhiBindlessDescriptorRef {
public:
  virtual void addUniformBuffer(RhiMultiBuffer *buffer, u32 loc) = 0;
  virtual void addStorageBuffer(RhiMultiBuffer *buffer, u32 loc) = 0;
  virtual void addStorageBuffer(RhiBuffer *buffer, u32 loc) = 0;
  virtual void addCombinedImageSampler(RhiTexture *texture, RhiSampler *sampler, u32 loc) = 0;
  virtual void addUAVImage(RhiTexture *texture, RhiImageSubResource subResource, u32 loc) = 0;
};

} // namespace Ifrit::GraphicsBackend::Rhi