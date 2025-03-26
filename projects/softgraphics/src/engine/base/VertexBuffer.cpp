
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

#include "ifrit/softgraphics/engine/base/VertexBuffer.h"
#include "ifrit/softgraphics/engine/base/TypeDescriptor.h"

namespace Ifrit::Graphics::SoftGraphics
{
	IFRIT_APIDECL VertexBuffer::VertexBuffer()
	{
		this->context = new std::remove_pointer_t<decltype(this->context)>();
	}
	IFRIT_APIDECL VertexBuffer::~VertexBuffer()
	{
		delete this->context;
	}

	IFRIT_APIDECL void VertexBuffer::allocateBuffer(const size_t numVertices)
	{
		int elementSizeX = 0;
		for (int i = 0; i < context->layout.size(); i++)
		{
			elementSizeX += context->layout[i].size;
		}
		context->buffer.resize(numVertices * elementSizeX);
		this->vertexCount = static_cast<int>(numVertices);
	}

	IFRIT_APIDECL void
	VertexBuffer::setLayout(const std::vector<TypeDescriptor>& layout)
	{
		this->context->layout = layout;
		this->context->offsets.resize(layout.size());
		int offset = 0;
		for (int i = 0; i < layout.size(); i++)
		{
			context->offsets[i] = offset;
			offset += layout[i].size;
			if (layout[i].type == TypeDescriptorEnum::IFTP_UNDEFINED)
			{
				printf("Undefined layout %d\n", layout[i].type);
				std::abort();
			}
		}
		elementSize = offset;
	}

	IFRIT_APIDECL void VertexBuffer::setVertexCount(const int vcnt)
	{
		this->vertexCount = vcnt;
	}

	IFRIT_APIDECL int VertexBuffer::getVertexCount() const
	{
		return vertexCount;
	}

	IFRIT_APIDECL int VertexBuffer::getAttributeCount() const
	{
		return static_cast<int>(context->layout.size());
	}

	IFRIT_APIDECL TypeDescriptor
	VertexBuffer::getAttributeDescriptor(int index) const
	{
		return context->layout[index];
	}

	/* DLL Compatible */
	IFRIT_APIDECL void
	VertexBuffer::setLayoutCompatible(const TypeDescriptor* layouts, int num)
	{
		std::vector<TypeDescriptor> clayouts(num);
		for (int i = 0; i < num; i++)
		{
			clayouts[i] = layouts[i];
		}
		setLayout(clayouts);
	}
	IFRIT_APIDECL void
	VertexBuffer::setValueFloat4Compatible(const int index, const int attribute,
		const Vector4f value)
	{
		this->setValue<Vector4f>(index, attribute, value);
	}

} // namespace Ifrit::Graphics::SoftGraphics