#include "engine/base/VertexShaderResult.h"

namespace Ifrit::Engine {

	IFRIT_APIDECL VertexShaderResult::VertexShaderResult(uint32_t vertexCount, uint32_t varyingCount) {
		this->vertexCount = vertexCount;
		this->varyings.resize(varyingCount);
	}

	IFRIT_APIDECL ifloat4* VertexShaderResult::getPositionBuffer() {
		return position.data();
	}
	IFRIT_APIDECL void VertexShaderResult::initializeVaryingBufferFromShader(const TypeDescriptor& typeDescriptor, int id) {
		this->varyings[id].resize(vertexCount * typeDescriptor.size);
		this->varyingDescriptors[id] = typeDescriptor;
	}
	IFRIT_APIDECL void VertexShaderResult::setVertexCount(const uint32_t vertexCount){
		this->vertexCount = vertexCount;
		for (auto& varying : varyings) {
			varying.resize(vertexCount * sizeof(ifloat4));
			varyingDescriptors.resize(varyings.size());
		}
		position.resize(vertexCount);
	}
}