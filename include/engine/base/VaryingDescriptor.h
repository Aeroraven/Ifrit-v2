#pragma once	

#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShaderResult.h"
#include "core/definition/CoreExports.h"
#include "engine/base/TypeDescriptor.h"

namespace Ifrit::Engine {
	class IFRIT_APIDECL VaryingDescriptor {
	protected:
		std::vector<TypeDescriptor> varyingDescriptors;
	public:
		void setVaryingDescriptors(const std::vector<TypeDescriptor>& varyingDescriptors);
		void applyVaryingDescriptors(VertexShaderResult* varyingBuffer);
		uint32_t getVaryingCounts() const { return varyingDescriptors.size(); }
		TypeDescriptor getVaryingDescriptor(int index) const { return varyingDescriptors[index]; }
	};
}