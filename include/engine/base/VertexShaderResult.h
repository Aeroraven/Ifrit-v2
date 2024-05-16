#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/TypeDescriptor.h"
#include "engine/base/VaryingStore.h"

namespace Ifrit::Engine {

	class VertexShaderResult {
	private:
		std::vector<ifloat4> position;
		std::vector<std::vector<VaryingStore>> varyings;
		std::vector<TypeDescriptor> varyingDescriptors;
		uint32_t vertexCount;
		uint32_t rawCounter = 0;

	public:
		VertexShaderResult(uint32_t vertexCount, uint32_t varyingCount);
		
		void initializeVaryingBuffer(int id) {
			varyings[id].resize(vertexCount);
		}

		inline VaryingStore* getVaryingBuffer(int id) {
			return varyings[id].data();
		}

		ifloat4* getPositionBuffer();
		void initializeVaryingBufferFromShader(const TypeDescriptor& typeDescriptor,int id);
		void setVertexCount(const uint32_t vertexCount);
		TypeDescriptor getVaryingDescriptor(int id) const { return varyingDescriptors[id]; }
		void allocateVaryings(int count) { varyings.resize(count); varyingDescriptors.resize(count); }

	};
}