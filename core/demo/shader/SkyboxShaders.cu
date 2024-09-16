#ifdef IFRIT_FEATURE_CUDA
#include "SkyboxShaders.cuh"
#include "core/cuda/CudaUtils.cuh"
#include "engine/math/ShaderBuiltinCuda.cuh"

namespace Ifrit::Demo::Skybox {
	IFRIT_DUAL void SkyboxVS::execute(const void* const* input, ifloat4* outPos, ifloat4* const* outVaryings) {
		using namespace Ifrit::Engine::Math::ShaderOps::CUDA;
		auto s = isbReadFloat4(input[0]);
		float4x4 view = (lookAt({ 0.0,0.0,0.0 }, { 0.0,0.0,-1.0 }, { 0,1.0,0 }));  
		float4x4 proj = (perspective(30 * 3.14159 / 180, 2048.0 / 1152.0, 0.001, 1000));
		float4x4 mvp = multiply(proj, view);
		auto p = multiply(mvp, s);
		*outPos = p;
		//printf("%f %f %f %f -> %f %f %f %f \n", s.x, s.y, s.z, s.w, p.x, p.y, p.z, p.w);
		*outVaryings[0] = isbReadFloat4(input[1]);
	}

	IFRIT_HOST Ifrit::Engine::VertexShader* SkyboxVS::getCudaClone() {
		return Ifrit::Core::CUDA::hostGetDeviceObjectCopy<SkyboxVS>(this);
	}

	IFRIT_DUAL void SkyboxFS::execute(const  void* varyings, void* colorOutput, float* fragmentDepth) {
		using Ifrit::Engine::Math::ShaderOps::CUDA::textureCubeLod;

		auto r = isbcuReadPsVarying(varyings, 0);
		auto& co = isbcuReadPsColorOut(colorOutput, 0);
		float3 uvw = { r.x,r.y,r.z };
		auto cl = textureCubeLod(0, 0, uvw, 0);

		co.x = cl.x;
		co.y = cl.y;
		co.z = cl.z;
		co.w = cl.w;
	}

	IFRIT_HOST Ifrit::Engine::FragmentShader* SkyboxFS::getCudaClone() {
		return Ifrit::Core::CUDA::hostGetDeviceObjectCopy<SkyboxFS>(this);
	}
}

#endif