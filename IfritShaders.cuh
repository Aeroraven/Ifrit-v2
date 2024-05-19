#pragma once
#include "engine/base/VertexShader.h"
#include "engine/base/FragmentShader.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/tilerastercuda/TileRasterInvocationCuda.cuh"

class DemoVertexShaderCuda : public  Ifrit::Engine::VertexShader {
public:
	IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, Ifrit::Engine::VaryingStore** outVaryings);
	IFRIT_HOST virtual Ifrit::Engine::VertexShader* getCudaClone() override;
};

class DemoFragmentShaderCuda : public  Ifrit::Engine::FragmentShader {
public:
	IFRIT_DUAL virtual void execute(const  Ifrit::Engine::VaryingStore* varyings, ifloat4* colorOutput);
	IFRIT_HOST virtual Ifrit::Engine::FragmentShader* getCudaClone() override;
};
