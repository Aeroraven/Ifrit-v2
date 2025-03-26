
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/softgraphics/core/utility/CoreUtils.h"
#include "ifrit/softgraphics/engine/base/TypeDescriptor.h"

namespace Ifrit::Graphics::SoftGraphics
{

    struct VertexBufferContext
    {
        std::vector<uint8_t>        buffer;
        std::vector<TypeDescriptor> layout;
        std::vector<int>            offsets;
    };

    class IFRIT_APIDECL VertexBuffer
    {
    private:
        VertexBufferContext* context;
        int                  vertexCount;
        int                  elementSize;

    public:
        VertexBuffer();
        ~VertexBuffer();
        void           setLayout(const std::vector<TypeDescriptor>& layout);
        void           allocateBuffer(const size_t numVertices);
        void           setVertexCount(const int vertexCount);
        int            getVertexCount() const;
        int            getAttributeCount() const;
        TypeDescriptor getAttributeDescriptor(int index) const;

        inline int     getOffset(int i) const { return context->offsets[i]; }
        inline int     getElementSize() const { return elementSize; }

        /* Templates */
        template <class T>
        inline T getValue(const int index, const int attribute) const
        {
            size_t      dOffset = context->offsets[attribute] + index * elementSize;
            const char* data    = reinterpret_cast<const char*>(&context->buffer[dOffset]);
            return *reinterpret_cast<const T*>(data);
        }

        template <class T>
        inline const T* getValuePtr(const int index, const int attribute) const
        {
            size_t      dOffset = context->offsets[attribute] + index * elementSize;
            const char* data    = reinterpret_cast<const char*>(&context->buffer[dOffset]);
            return reinterpret_cast<const T*>(data);
        }

        template <class T>
        inline T setValue(const int index, const int attribute, const T value)
        {
            size_t dOffset              = context->offsets[attribute] + index * elementSize;
            char*  data                 = reinterpret_cast<char*>(&context->buffer[dOffset]);
            *reinterpret_cast<T*>(data) = value;
            return value;
        }

        /* Inline */
        inline char* GetBufferUnsafe() const { return (char*)context->buffer.data(); }

        inline u32   GetBufferSize() const
        {
            using namespace Ifrit::Common::Utility;
            return SizeCast<int>(context->buffer.size());
        }

        /* DLL Compatible */
        void setLayoutCompatible(const TypeDescriptor* layouts, int num);
        void setValueFloat4Compatible(const int index, const int attribute, const Vector4f value);
    };
} // namespace Ifrit::Graphics::SoftGraphics