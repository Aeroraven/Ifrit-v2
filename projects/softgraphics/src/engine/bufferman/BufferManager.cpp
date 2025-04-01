
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

#include "ifrit/softgraphics/engine/bufferman/BufferManager.h"
#include "ifrit/core/typing/Util.h"
using namespace Ifrit;

namespace Ifrit::Graphics::SoftGraphics::BufferManager::Impl
{
    int BufferManagerImpl::allocateBufferId()
    {
        if (freeBufferIds.empty())
        {
            this->bufferMetadata.resize(this->bufferMetadata.size() + 1);
            this->buffers.resize(this->buffers.size() + 1);
            return SizeCast<int>(this->buffers.size()) - 1;
        }
        else
        {
            int id = freeBufferIds.top();
            freeBufferIds.pop();
            return id;
        }
    }
    BufferManagerImpl::BufferManagerImpl(std::shared_ptr<TrivialBufferManager> wrapperObject)
    {
        this->wrapperObject = wrapperObject;
    }
    BufferManagerImpl::~BufferManagerImpl() = default;

    IfritBuffer BufferManagerImpl::CreateBuffer(const IfritBufferCreateInfo& pCI)
    {
        auto bufferId                       = allocateBufferId();
        buffers[bufferId].id                = bufferId;
        buffers[bufferId].manager           = this->wrapperObject;
        bufferMetadata[bufferId].size       = pCI.bufferSize;
        bufferMetadata[bufferId].data       = std::make_unique<char[]>(pCI.bufferSize);
        bufferMetadata[bufferId].maintained = true;
        return buffers[bufferId];
    }

    void BufferManagerImpl::destroyBuffer(const IfritBuffer& buffer)
    {
        auto maintained = bufferMetadata[buffer.id].maintained;
        if (maintained && buffer.id >= 0 && !buffer.manager.owner_before(wrapperObject)
            && !wrapperObject.owner_before(buffer.manager))
        {
            freeBufferIds.push(buffer.id);
            bufferMetadata[buffer.id].size       = 0;
            bufferMetadata[buffer.id].data       = nullptr;
            bufferMetadata[buffer.id].maintained = false;
        }
        else
        {
            throw std::runtime_error("Buffer does not belong to this manager");
        }
    }

    void BufferManagerImpl::mapBufferMemory(const IfritBuffer& buffer, void** ppData)
    {
        if (!bufferMetadata[buffer.id].maintained)
        {
            throw std::runtime_error("Buffer is not maintained by this manager");
        }
        *ppData = bufferMetadata[buffer.id].data.get();
    }

    void BufferManagerImpl::bufferData(const IfritBuffer& buffer, const void* src, size_t offset, size_t size)
    {
        if (!bufferMetadata[buffer.id].maintained)
        {
            throw std::runtime_error("Buffer is not maintained by this manager");
        }
        if (offset + size > bufferMetadata[buffer.id].size)
        {
            throw std::runtime_error("Buffer overflow");
        }
        memcpy(bufferMetadata[buffer.id].data.get() + offset, src, size);
    }
    void BufferManagerImpl::bufferDataUnsafe(
        const IfritBuffer& buffer, const void* src, size_t offset, size_t size) IFRIT_AP_NOTHROW
    {
        memcpy(bufferMetadata[buffer.id].data.get() + offset, src, size);
    }
} // namespace Ifrit::Graphics::SoftGraphics::BufferManager::Impl

namespace Ifrit::Graphics::SoftGraphics::BufferManager
{
    TrivialBufferManager::TrivialBufferManager() {}
    TrivialBufferManager::~TrivialBufferManager() {}
    void TrivialBufferManager::Init()
    {
        initialized = true;
        impl        = std::make_unique<Impl::BufferManagerImpl>(shared_from_this());
    }
    IfritBuffer TrivialBufferManager::CreateBuffer(const IfritBufferCreateInfo& pCI)
    {
        if (!initialized)
        {
            ifritError("Buffer manager not initialized");
        }
        return impl->CreateBuffer(pCI);
    }
    void TrivialBufferManager::destroyBuffer(const IfritBuffer& buffer)
    {
        if (!initialized)
        {
            ifritError("Buffer manager not initialized");
        }
        return impl->destroyBuffer(buffer);
    }
    void TrivialBufferManager::mapBufferMemory(const IfritBuffer& buffer, void** ppData)
    {
        if (!initialized)
        {
            ifritError("Buffer manager not initialized");
        }
        return impl->mapBufferMemory(buffer, ppData);
    }
    void TrivialBufferManager::bufferData(const IfritBuffer& buffer, const void* src, size_t offset, size_t size)
    {
        if (!initialized)
        {
            ifritError("Buffer manager not initialized");
        }
        return impl->bufferData(buffer, src, offset, size);
    }
} // namespace Ifrit::Graphics::SoftGraphics::BufferManager