#include "engine/base/VertexShaderResult.h"

namespace Ifrit::Engine {

	VertexShaderResult::VertexShaderResult(uint32_t vertexCount, uint32_t varyingCount) {
		this->vertexCount = vertexCount;
		this->varyings.resize(varyingCount);
	}

	float4* VertexShaderResult::getPositionBuffer() {
		return position.data();
	}
	void VertexShaderResult::initializeVaryingBufferFromShader(const TypeDescriptor& typeDescriptor, int id) {
		this->varyings[id].resize(vertexCount * typeDescriptor.size);
		this->varyingDescriptors[id] = typeDescriptor;
	}
	void VertexShaderResult::setVertexCount(const uint32_t vertexCount){
		this->vertexCount = vertexCount;
		for (auto& varying : varyings) {
			varying.resize(vertexCount * sizeof(float4));
			varyingDescriptors.resize(varyings.size());
		}
		position.resize(vertexCount);
	}
}