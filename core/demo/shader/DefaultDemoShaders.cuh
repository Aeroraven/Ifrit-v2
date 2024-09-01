#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "engine/base/Shaders.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/math/ShaderBuiltin.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"


namespace Ifrit::Demo::DemoDefault {
	class DemoVertexShaderCuda : public  Ifrit::Engine::VertexShader {
	public:
		IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, Ifrit::Engine::VaryingStore* const* outVaryings) override;
		IFRIT_HOST virtual Ifrit::Engine::VertexShader* getCudaClone() override;
	};

	class DemoFragmentShaderCuda : public  Ifrit::Engine::FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const  void* varyings, void* colorOutput, float* fragmentDepth) override;
		IFRIT_HOST virtual Ifrit::Engine::FragmentShader* getCudaClone() override;
	};

	class DemoGeometryShaderCuda : public Ifrit::Engine::GeometryShader {
	public:
		uint32_t atMaxVertices = 4;
		IFRIT_DUAL virtual void execute(
			const ifloat4* const* inPos,
			const Ifrit::Engine::VaryingStore* const* inVaryings,
			ifloat4* outPos,
			Ifrit::Engine::VaryingStore* outVaryings,
			int* outSize
		) override;
		IFRIT_HOST virtual GeometryShader* getCudaClone();
	};
}
#endif