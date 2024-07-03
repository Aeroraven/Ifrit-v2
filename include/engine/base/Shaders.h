#pragma once

#include "engine/base/VaryingStore.h"


namespace Ifrit::Engine {
	enum GeometryShaderTopology {
		IGST_TRIANGLES = 0,
		IGST_LINES = 1,
		IGST_POINTS = 2
	};

	class VertexShader {
	public:
		IFRIT_DUAL virtual void execute(
			const void* const* input,
			ifloat4* outPos,
			VaryingStore** outVaryings
		) {};
		IFRIT_HOST virtual VertexShader* getCudaClone() { return nullptr; };
	};

	class FragmentShader {
	public:
		float* atTexture[32];
		uint32_t atTextureWid[32];
		uint32_t atTextureHei[32];
		IFRIT_DUAL virtual void execute(
			const void* varyings, 
			void* colorOutput
		) = 0;
		IFRIT_HOST virtual FragmentShader* getCudaClone() { return nullptr; };
	};

	class GeometryShader {
	public:
		GeometryShaderTopology atTopology = IGST_TRIANGLES;
		uint32_t atMaxVertices = 3;
		IFRIT_DUAL virtual void execute(
			const ifloat4* const inPos,
			const VaryingStore* const* const inVaryings,
			ifloat4* outPos,
			VaryingStore** outVaryings
		) = 0;
		IFRIT_HOST virtual GeometryShader* getCudaClone() { return nullptr; };
	};
}