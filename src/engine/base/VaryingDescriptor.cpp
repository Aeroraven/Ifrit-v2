#include "engine/base/VaryingDescriptor.h"

namespace Ifrit::Engine {
	void VaryingDescriptor::setVaryingDescriptors(const std::vector<TypeDescriptor>& varyingDescriptors) {
		this->varyingDescriptors = varyingDescriptors;

	}
	void VaryingDescriptor::applyVaryingDescriptors(VertexShaderResult* varyingBuffer) {
		for (int i = 0; i < varyingDescriptors.size(); i++) {
			varyingBuffer->initializeVaryingBufferFromShader(varyingDescriptors[i], i);
		}
	}
}