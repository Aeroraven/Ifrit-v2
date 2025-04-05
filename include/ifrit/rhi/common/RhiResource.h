
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
#include "ifrit/core/logging/Logging.h"
#include <queue>
#include <cstddef>

namespace Ifrit::Graphics::Rhi
{

    // UPD 250325: Resource removal algo before destroys the resource that still in use on device side
    // referencing Unreal's resource state management, a delete queue should be maintained

    class IFRIT_APIDECL IRhiDeviceResourceDeleteQueue
    {
    public:
        virtual void AddResourceToDeleteQueue(RhiDeviceResource* resource) = 0;
        virtual i32  ProcessDeleteQueue()                                  = 0;
    };

    class IFRIT_APIDECL RhiDeviceResource
    {
    private:
        Atomic<u32>                    m_refCount = 0;
        IRhiDeviceResourceDeleteQueue* m_deleteQueue;
        RhiDescriptorHandle            m_descHandle;
        bool                           m_isUnmanaged = false; // unmanaged resources are EXTERNAL resources
        String                         m_debugName;

    public:
        explicit RhiDeviceResource(nullptr_t v)
            : m_deleteQueue(nullptr), m_descHandle(RhiDescriptorHeapType::Invalid, ~0u), m_isUnmanaged(true)
        {
        }
        RhiDeviceResource(IRhiDeviceResourceDeleteQueue* deleteQueue)
            : m_deleteQueue(deleteQueue), m_descHandle(RhiDescriptorHeapType::Invalid, ~0u)
        {
        }
        virtual ~RhiDeviceResource() {}

        inline virtual void AddRef() { m_refCount.fetch_add(1); }
        inline virtual void Release()
        {
            if (m_refCount.fetch_sub(1) == 1)
            {
                if (!m_isUnmanaged)
                {
                    MarkForDelete();
                }
            }
        }
        IF_FORCEINLINE virtual void MarkForDelete() { m_deleteQueue->AddResourceToDeleteQueue(this); }
        IF_FORCEINLINE virtual void SetDescriptorHandle(const RhiDescriptorHandle& handle) { m_descHandle = handle; }
        IF_FORCEINLINE virtual u32  GetDescId() const
        {
            if (m_descHandle.GetType() == RhiDescriptorHeapType::Invalid)
            {
                iError("Invalid descriptor handle");
                std::abort();
                return ~0u;
            }
            return m_descHandle.GetId();
        }
        IF_FORCEINLINE virtual void          SetDebugName(const String& name) { m_debugName = name; }
        IF_FORCEINLINE virtual const String& GetDebugName() const { return m_debugName; }
    };

    class IFRIT_APIDECL RhiBuffer : public RhiDeviceResource
    {
    protected:
        RhiDevice*       m_context;
        RhiResourceState m_state = RhiResourceState::Undefined;

    private:
        inline void SetState(RhiResourceState state) { m_state = state; }

    public:
        RhiBuffer(IRhiDeviceResourceDeleteQueue* deleteQueue) : RhiDeviceResource(deleteQueue) {}
        virtual ~RhiBuffer()                                                                = default;
        virtual void                    MapMemory()                                         = 0;
        virtual void                    UnmapMemory()                                       = 0;
        virtual void                    FlushBuffer()                                       = 0;
        virtual void                    ReadBuffer(void* data, u32 size, u32 offset)        = 0;
        virtual void                    WriteBuffer(const void* data, u32 size, u32 offset) = 0;
        virtual inline RhiResourceState GetState() const { return m_state; }

        virtual RhiDeviceAddr           GetDeviceAddress() const = 0;

        friend class RhiCommandList;
    };

    class IFRIT_APIDECL RhiMultiBuffer
    {
    protected:
        RhiDevice*                     m_context;
        IRhiDeviceResourceDeleteQueue* m_deleteQueue;

    public:
        RhiMultiBuffer(IRhiDeviceResourceDeleteQueue* deleteQueue) : m_deleteQueue(deleteQueue) {}
        virtual RhiBuffer* GetActiveBuffer()                       = 0;
        virtual RhiBuffer* GetActiveBufferRelative(u32 deltaFrame) = 0;
        virtual ~RhiMultiBuffer()                                  = default;
    };

    class IFRIT_APIDECL RhiStagedSingleBuffer
    {
    protected:
        RhiDevice* m_context;

    public:
        virtual ~RhiStagedSingleBuffer()                                                                     = default;
        virtual void CmdCopyToDevice(const RhiCommandList* cmd, const void* data, u32 size, u32 localOffset) = 0;
    };

    class RhiStagedMultiBuffer
    {
    };

    class IFRIT_APIDECL RhiTexture : public RhiDeviceResource
    {
    protected:
        RhiDevice*       m_context;
        RhiResourceState m_state             = RhiResourceState::Undefined;
        bool             m_rhiSwapchainImage = false;

    private:
        inline void SetState(RhiResourceState state) { m_state = state; }

    public:
        explicit RhiTexture(nullptr_t v) : RhiDeviceResource(nullptr) {}
        RhiTexture(IRhiDeviceResourceDeleteQueue* deleteQueue) : RhiDeviceResource(deleteQueue) {}
        virtual ~RhiTexture()                                  = default;
        virtual u32                     GetHeight() const      = 0;
        virtual u32                     GetWidth() const       = 0;
        virtual u32                     GetDepth() const       = 0;
        virtual bool                    IsDepthTexture() const = 0;
        virtual inline RhiResourceState GetState() const { return m_state; }
        virtual void*                   GetNativeHandle() const = 0;
        virtual u32                     GetSamples() const      = 0;

        friend class RhiCommandList;
    };

    class IFRIT_APIDECL RhiSampler : public RhiDeviceResource
    {
    protected:
        RhiSampler(IRhiDeviceResourceDeleteQueue* deleteQueue) : RhiDeviceResource(deleteQueue) {}
        virtual int _polymorphismPlaceHolder() { return 0; }
    };

    struct IFRIT_APIDECL RhiRTGeometryReference
    {
        RhiDeviceAddr m_vertex;
        RhiDeviceAddr m_index;
        RhiDeviceAddr m_transform;
        u32           m_numVertices;
        u32           m_numIndices;
        u32           m_vertexComponents = 3;
        u32           m_vertexStride     = 12;
    };

    class IFRIT_APIDECL RhiRTInstance
    {
    public:
        virtual RhiDeviceAddr GetDeviceAddress() const = 0;
    };

    class IFRIT_APIDECL RhiRTScene
    {
    public:
        virtual RhiDeviceAddr GetDeviceAddress() const = 0;
    };

    class IFRIT_APIDECL RhiBindlessDescriptorRef
    {
    public:
        virtual void AddUniformBuffer(RhiMultiBuffer* buffer, u32 loc)                          = 0;
        virtual void AddStorageBuffer(RhiMultiBuffer* buffer, u32 loc)                          = 0;
        virtual void AddStorageBuffer(RhiBuffer* buffer, u32 loc)                               = 0;
        virtual void AddCombinedImageSampler(RhiTexture* texture, RhiSampler* sampler, u32 loc) = 0;
        virtual void AddUAVImage(RhiTexture* texture, RhiImageSubResource subResource, u32 loc) = 0;
    };

} // namespace Ifrit::Graphics::Rhi