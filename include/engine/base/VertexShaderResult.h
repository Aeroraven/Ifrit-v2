#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Engine {

	class VertexShaderResult {
	private:
		std::vector<float4> position;
		std::vector<std::vector<char>> varyings;
		uint32_t vertexCount;

#ifdef IFRIT_VERBOSE_SAFETY_CHECK
		std::vector<uint8_t> varyingElementSize;
		std::vector<uint8_t> varyingIsIntegral;
		std::vector<uint8_t> varyingIsUnsigned;
#endif

	public:
		VertexShaderResult(uint32_t vertexCount, uint32_t varyingCount);
		
		template<class T>
		void initializeVaryingBuffer(int id) {
			varyings[id].resize(vertexCount * sizeof(T));

#ifdef IFRIT_VERBOSE_SAFETY_CHECK
			varyingElementSize[id] = sizeof(T);
			varyingIsIntegral[id] = std::is_integral<T>::value;
			varyingIsUnsigned[id] = std::is_unsigned<T>::value;
#endif

		}

		template<class T>
		T* getVaryingBuffer(int id) {

#ifdef IFRIT_VERBOSE_SAFETY_CHECK
			if (varyingElementSize[0] != sizeof(T)) {
				ifritError("Varying buffer element size does not match the requested type");
			}
			if (varyingIsIntegral[0] != std::is_integral<T>::value) {
				ifritError("Varying buffer element type is not integral");
			}
			if (varyingIsUnsigned[0] != std::is_unsigned<T>::value) {
				ifritError("Varying buffer element type is not unsigned");
			}
#endif
			return reinterpret_cast<T*>(varyings[id].data());
		}

		float4* getPositionBuffer();
	};
}