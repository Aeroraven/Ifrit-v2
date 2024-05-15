#include "engine/base/VertexShader.h"

namespace Ifrit::Engine {
	void VertexShader::setVaryingDescriptors(const std::vector<TypeDescriptor>& varyingDescriptors){
		this->varyingDescriptors = varyingDescriptors;
		
	}
	void VertexShader::applyVaryingDescriptors(VertexShaderResult* varyingBuffer){
		for (int i = 0; i < varyingDescriptors.size(); i++) {
			varyingBuffer->initializeVaryingBufferFromShader(varyingDescriptors[i], i);
		}
	}
}