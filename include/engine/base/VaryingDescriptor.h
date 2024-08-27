#pragma once	

#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShaderResult.h"
#include "core/definition/CoreExports.h"
#include "engine/base/TypeDescriptor.h"

namespace Ifrit::Engine {

	struct VaryingDescriptorContext {
		std::vector<TypeDescriptor> varyingDescriptors;
	};

	class IFRIT_APIDECL VaryingDescriptor {
	protected:
		VaryingDescriptorContext* context;
	public:
		VaryingDescriptor();
		~VaryingDescriptor();
		void setVaryingDescriptors(const std::vector<TypeDescriptor>& varyingDescriptors);
		void applyVaryingDescriptors(VertexShaderResult* varyingBuffer);

		/* Inline */
		inline uint32_t getVaryingCounts() const {
			return context->varyingDescriptors.size();
		}
		inline TypeDescriptor getVaryingDescriptor(int index) const { 
			return context->varyingDescriptors[index];
		}

		/* DLL Compat */
		void setVaryingDescriptorsCompatible(const TypeDescriptor* varyingDescriptors, int num);
	};
}