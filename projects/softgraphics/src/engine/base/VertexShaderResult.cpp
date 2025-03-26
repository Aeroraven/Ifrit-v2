
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

#include "ifrit/softgraphics/engine/base/VertexShaderResult.h"

namespace Ifrit::Graphics::SoftGraphics
{

	IFRIT_APIDECL VertexShaderResult::VertexShaderResult(uint32_t vertexCount,
		uint32_t												  varyingCount)
	{
		this->context = new std::remove_pointer_t<decltype(this->context)>();
		this->vertexCount = vertexCount;
		this->context->varyings.resize(varyingCount);
	}
	IFRIT_APIDECL VertexShaderResult::~VertexShaderResult()
	{
		delete this->context;
	}

	IFRIT_APIDECL Vector4f* VertexShaderResult::getPositionBuffer()
	{
		return context->position.data();
	}
	IFRIT_APIDECL void VertexShaderResult::initializeVaryingBufferFromShader(
		const TypeDescriptor& typeDescriptor, int id)
	{
		this->context->varyings[id].resize(vertexCount * typeDescriptor.size);
		this->context->varyingDescriptors[id] = typeDescriptor;
	}
	IFRIT_APIDECL void VertexShaderResult::setVertexCount(const uint32_t vcnt)
	{
		this->vertexCount = vcnt;
		for (auto& varying : context->varyings)
		{
			varying.resize(vertexCount * sizeof(Vector4f));
			context->varyingDescriptors.resize(context->varyings.size());
		}
		context->position.resize(vertexCount);
	}
} // namespace Ifrit::Graphics::SoftGraphics