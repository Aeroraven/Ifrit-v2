#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "engine/base/Shaders.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/math/ShaderBuiltin.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"


namespace Ifrit::Demo::MeshletDemo {
	class MeshletDemoCuMS : public  Ifrit::Engine::SoftRenderer::MeshShader {
	public:
		IFRIT_DUAL virtual void execute(
			iint3 localInvocation,
			int workGroupId,
			const void* inTaskShaderPayload,
			Ifrit::Engine::SoftRenderer::VaryingStore* outVaryings,
			ifloat4* outPos,
			int* outIndices,
			int& outNumVertices,
			int& outNumIndices
		) override;
		IFRIT_HOST virtual Ifrit::Engine::SoftRenderer::MeshShader* getCudaClone() override;
	};

	class MeshletDemoCuTS : public  Ifrit::Engine::SoftRenderer::TaskShader {
	public:
		IFRIT_DUAL virtual void execute(
			int workGroupId,
			void* outTaskShaderPayload,
			iint3* outMeshWorkGroups,
			int& outNumMeshWorkGroups
		);
		IFRIT_HOST virtual Ifrit::Engine::SoftRenderer::TaskShader* getCudaClone() override;
	};

	class MeshletDemoCuFS : public  Ifrit::Engine::SoftRenderer::FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const  void* varyings, void* colorOutput, float* fragmentDepth) override;
		IFRIT_HOST virtual Ifrit::Engine::SoftRenderer::FragmentShader* getCudaClone() override;
	};
}
#endif