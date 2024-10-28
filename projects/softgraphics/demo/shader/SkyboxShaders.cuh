#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "engine/base/Shaders.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/math/ShaderBuiltin.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"


namespace Ifrit::Demo::Skybox {
	class SkyboxVS : public  Ifrit::Engine::SoftRenderer::VertexShader {
	public:
		IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, ifloat4* const* outVaryings) override;
		IFRIT_HOST virtual Ifrit::Engine::SoftRenderer::VertexShader* getCudaClone() override;
	};

	class SkyboxFS : public  Ifrit::Engine::SoftRenderer::FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const  void* varyings, void* colorOutput, float* fragmentDepth);
		IFRIT_HOST virtual Ifrit::Engine::SoftRenderer::FragmentShader* getCudaClone() override;
	};
}
#endif