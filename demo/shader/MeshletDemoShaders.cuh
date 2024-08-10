#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "engine/base/Shaders.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/math/ShaderBuiltin.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"


namespace Ifrit::Demo::MeshletDemo {
	class MeshletDemoCuMS : public  Ifrit::Engine::MeshShader {
	public:
		IFRIT_DUAL virtual void execute(
			iint3 localInvocation,
			int workGroupId,
			Ifrit::Engine::VaryingStore* outVaryings,
			ifloat4* outPos,
			int* outIndices,
			int& outNumVertices,
			int& outNumIndices
		);
		IFRIT_HOST virtual Ifrit::Engine::MeshShader* getCudaClone() override;
	};

	class MeshletDemoCuFS : public  Ifrit::Engine::FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const  void* varyings, void* colorOutput);
		IFRIT_HOST virtual Ifrit::Engine::FragmentShader* getCudaClone() override;
	};
}
#endif