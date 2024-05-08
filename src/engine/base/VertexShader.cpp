#include "engine/base/VertexShader.h"

namespace Ifrit::Engine {
	void VertexShader::bindVertexBuffer(VertexBuffer& vertexBuffer) {
		this->vertexBuffer = &vertexBuffer;
	}
	void VertexShader::bindVaryingBuffer(VertexShaderResult& varyingBuffer) {
		this->varyingBuffer = &varyingBuffer;
	}
}