#pragma once
#include "engine/base/Shaders.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/math/ShaderBuiltin.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"

class DemoVertexShaderCuda : public  Ifrit::Engine::VertexShader {
public:
	IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, Ifrit::Engine::VaryingStore** outVaryings);
	IFRIT_HOST virtual Ifrit::Engine::VertexShader* getCudaClone() override;
};

class DemoFragmentShaderCuda : public  Ifrit::Engine::FragmentShader {
public:
	IFRIT_DUAL virtual void execute(const  void* varyings, void* colorOutput);
	IFRIT_HOST virtual Ifrit::Engine::FragmentShader* getCudaClone() override;
};

class DemoGeometryShaderCuda : public Ifrit::Engine::GeometryShader {
public:
	uint32_t atMaxVertices = 4;
	IFRIT_DUAL virtual void execute(
		const ifloat4** inPos,
		const Ifrit::Engine::VaryingStore** inVaryings,
		ifloat4* outPos,
		Ifrit::Engine::VaryingStore* outVaryings,
		int* outSize
	);
	IFRIT_HOST virtual GeometryShader* getCudaClone();
};