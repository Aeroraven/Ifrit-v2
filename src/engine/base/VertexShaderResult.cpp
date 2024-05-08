#include "engine/base/VertexShaderResult.h"

namespace Ifrit::Engine {

	VertexShaderResult::VertexShaderResult(uint32_t vertexCount, uint32_t varyingCount) {
		this->vertexCount = vertexCount;
		this->varyings.resize(varyingCount);
	}

	float4* VertexShaderResult::getPositionBuffer() {
		return position.data();
	}
}