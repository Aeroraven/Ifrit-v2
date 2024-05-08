#pragma once
#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShaderResult.h"
namespace Ifrit::Engine {
	class VertexShader{
	private:
		VertexBuffer* vertexBuffer;
		VertexShaderResult* varyingBuffer;
	public:
		void bindVertexBuffer(VertexBuffer& vertexBuffer);
		void bindVaryingBuffer(VertexShaderResult& varyingBuffer);
		virtual void execute(const int id) = 0;
	};
}