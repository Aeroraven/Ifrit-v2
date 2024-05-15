#pragma once
#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShaderResult.h"
#include "core/definition/CoreExports.h"
#include "engine/base/TypeDescriptor.h"

namespace Ifrit::Engine {
	class VertexShader{
	protected:
		std::vector<TypeDescriptor> varyingDescriptors;
	public:
		void setVaryingDescriptors(const std::vector<TypeDescriptor>& varyingDescriptors);
		void applyVaryingDescriptors(VertexShaderResult* varyingBuffer);
		IFRIT_DUAL virtual void execute(const void* const* input, float4* outPos, VaryingStore** outVaryings) {};
		uint32_t getVaryingCounts() const { return varyingDescriptors.size(); }
	};
}