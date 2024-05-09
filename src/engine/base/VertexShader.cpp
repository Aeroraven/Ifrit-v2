#include "engine/base/VertexShader.h"

namespace Ifrit::Engine {
	void VertexShader::setVaryingDescriptors(const std::vector<TypeDescriptor>& varyingDescriptors){
		this->varyingDescriptors = varyingDescriptors;
		
	}
	void VertexShader::applyVaryingDescriptors(){
		for (int i = 0; i < varyingDescriptors.size(); i++) {
			varyingBuffer->initializeVaryingBufferFromShader(varyingDescriptors[i], i);
		}
	}
	void VertexShader::bindVertexBuffer(const VertexBuffer& vertexBuffer) {
		this->vertexBuffer = &vertexBuffer;
	}
	void VertexShader::bindVaryingBuffer(VertexShaderResult& varyingBuffer) {
		this->varyingBuffer = &varyingBuffer;
	}
}