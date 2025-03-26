
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
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/Structures.h"
#include <stack>

namespace Ifrit::Graphics::SoftGraphics::BufferManager
{
    class TrivialBufferManager;

    struct IfritBuffer
    {
        int                                 id = -1;
        std::weak_ptr<TrivialBufferManager> manager;
    };

    struct IfritBufferMetadata
    {
        std::unique_ptr<char[]> data       = nullptr;
        size_t                  size       = 0;
        bool                    maintained = false;
    };

    namespace Impl
    {
        class BufferManagerImpl
        {
        private:
            std::weak_ptr<TrivialBufferManager> wrapperObject;
            std::vector<IfritBuffer>            buffers;
            std::vector<IfritBufferMetadata>    bufferMetadata;
            std::stack<int>                     freeBufferIds;

        protected:
            int allocateBufferId();

        public:
            BufferManagerImpl(std::shared_ptr<TrivialBufferManager> wrapperObject);
            ~BufferManagerImpl();

            IfritBuffer CreateBuffer(const IfritBufferCreateInfo& pCI);
            void        destroyBuffer(const IfritBuffer& buffer);
            void        mapBufferMemory(const IfritBuffer& buffer, void** ppData);
            void        bufferData(const IfritBuffer& buffer, const void* src, size_t offset,
                       size_t size);
            void        bufferDataUnsafe(const IfritBuffer& buffer, const void* src,
                       size_t offset, size_t size) IFRIT_AP_NOTHROW;
        };
    } // namespace Impl

    class IFRIT_APIDECL TrivialBufferManager : public std::enable_shared_from_this<TrivialBufferManager>
    {
    private:
        bool                                     initialized = false;
        std::unique_ptr<Impl::BufferManagerImpl> impl;

    public:
        TrivialBufferManager();
        ~TrivialBufferManager();
        void        Init();
        IfritBuffer CreateBuffer(const IfritBufferCreateInfo& pCI);
        void        destroyBuffer(const IfritBuffer& buffer);
        void        mapBufferMemory(const IfritBuffer& buffer, void** ppData);
        void        bufferData(const IfritBuffer& buffer, const void* src, size_t offset,
                   size_t size);
    };
} // namespace Ifrit::Graphics::SoftGraphics::BufferManager