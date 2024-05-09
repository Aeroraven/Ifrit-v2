#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/TypeDescriptor.h"

namespace Ifrit::Engine {

	class VertexShaderResult {
	private:
		std::vector<float4> position;
		std::vector<std::vector<char>> varyings;
		uint32_t vertexCount;
		uint32_t rawCounter = 0;

	public:
		VertexShaderResult(uint32_t vertexCount, uint32_t varyingCount);
		
		template<class T>
		void initializeVaryingBuffer(int id) {
			varyings[id].resize(vertexCount * sizeof(T));
		}

		template<class T>
		T* getVaryingBuffer(int id) {
			return reinterpret_cast<T*>(varyings[id].data());
		}

		float4* getPositionBuffer();
		void initializeVaryingBufferFromShader(const TypeDescriptor& typeDescriptor,int id);
		void setVertexCount(const uint32_t vertexCount);

	};
}