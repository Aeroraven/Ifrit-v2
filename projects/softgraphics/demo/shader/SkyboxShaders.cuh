#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "engine/base/Shaders.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/math/ShaderBuiltin.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"


namespace Ifrit::Demo::Skybox {
	class SkyboxVS : public  Ifrit::SoftRenderer::VertexShader {
	public:
		IFRIT_DUAL virtual void execute(const void* const* input, Vector4f* outPos, Vector4f* const* outVaryings) override;
		IFRIT_HOST virtual Ifrit::SoftRenderer::VertexShader* getCudaClone() override;
	};

	class SkyboxFS : public  Ifrit::SoftRenderer::FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const  void* varyings, void* colorOutput, float* fragmentDepth);
		IFRIT_HOST virtual Ifrit::SoftRenderer::FragmentShader* getCudaClone() override;
	};
}
#endif