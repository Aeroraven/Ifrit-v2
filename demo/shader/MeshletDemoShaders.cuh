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
			const void* inTaskShaderPayload,
			Ifrit::Engine::VaryingStore* outVaryings,
			ifloat4* outPos,
			int* outIndices,
			int& outNumVertices,
			int& outNumIndices
		) override;
		IFRIT_HOST virtual Ifrit::Engine::MeshShader* getCudaClone() override;
	};

	class MeshletDemoCuTS : public  Ifrit::Engine::TaskShader {
	public:
		IFRIT_DUAL virtual void execute(
			int workGroupId,
			void* outTaskShaderPayload,
			iint3* outMeshWorkGroups,
			int& outNumMeshWorkGroups
		);
		IFRIT_HOST virtual Ifrit::Engine::TaskShader* getCudaClone() override;
	};

	class MeshletDemoCuFS : public  Ifrit::Engine::FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const  void* varyings, void* colorOutput, float& fragmentDepth) override;
		IFRIT_HOST virtual Ifrit::Engine::FragmentShader* getCudaClone() override;
	};
}
#endif