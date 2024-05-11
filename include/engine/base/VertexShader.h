#pragma once
#include "engine/base/VertexBuffer.h"
#include "engine/base/VertexShaderResult.h"
#include "core/definition/CoreExports.h"
#include "engine/base/TypeDescriptor.h"

namespace Ifrit::Engine {
	class VertexShader{
	protected:
		const VertexBuffer* vertexBuffer;
		VertexShaderResult* varyingBuffer;
		std::vector<TypeDescriptor> varyingDescriptors;
	public:
		void setVaryingDescriptors(const std::vector<TypeDescriptor>& varyingDescriptors);
		void applyVaryingDescriptors();
		void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		void bindVaryingBuffer(VertexShaderResult& varyingBuffer);
		virtual void execute(const std::vector<const void*>& input, float4& outPos, std::vector<VaryingStore*>& outVaryings) {};
		uint32_t getVaryingCounts() const { return varyingDescriptors.size(); }
	};
}