#include "engine/base/VaryingDescriptor.h"

namespace Ifrit::Engine {
	IFRIT_APIDECL void VaryingDescriptor::setVaryingDescriptors(const std::vector<TypeDescriptor>& varyingDescriptors) {
		this->varyingDescriptors = varyingDescriptors;

	}
	IFRIT_APIDECL void VaryingDescriptor::applyVaryingDescriptors(VertexShaderResult* varyingBuffer) {
		for (int i = 0; i < varyingDescriptors.size(); i++) {
			varyingBuffer->initializeVaryingBufferFromShader(varyingDescriptors[i], i);
		}
	}
}