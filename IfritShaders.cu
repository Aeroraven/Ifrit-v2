#include "IfritShaders.cuh"
#include "core/cuda/CudaUtils.cuh"

IFRIT_DUAL void DemoVertexShaderCuda::execute(const void* const* input, ifloat4* outPos, Ifrit::Engine::VaryingStore** outVaryings) {
	using namespace Ifrit::Engine::Math::ShaderOps::CUDA;
	//float4x4 view = (lookAt({ 0,1.5,5.25 }, { 0,1.5,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 0,0.75,1.50 }, { 0,0.75,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 0,0.1,1.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
	float4x4 view = (lookAt({ 0.1,0.05,0.1 }, { 0,0.05,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 500,300,0 }, { -100,300,-0 }, { 0,1,0 }));
	//float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 10.0, 3000));

	//float4x4 view = (lookAt({ 0,1.5,0}, { -100,1.5,0 }, { 0,1,0 }));
	float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 1.0, 1000));
	float4x4 mvp = multiply(proj, view);
	auto s = *reinterpret_cast<const float4*>(input[0]);
	auto p = multiply(mvp, s);
	*(float4*)outPos = p;

	outVaryings[0]->vf4 = *reinterpret_cast<const ifloat4*>(input[1]);
	outVaryings[1]->vf4 = *reinterpret_cast<const ifloat4*>(input[2]);
}
IFRIT_HOST Ifrit::Engine::VertexShader* DemoVertexShaderCuda::getCudaClone() {
	return Ifrit::Core::CUDA::hostGetDeviceObjectCopy<DemoVertexShaderCuda>(this);
}

IFRIT_DUAL void DemoFragmentShaderCuda::execute(const  void* varyings, void* colorOutput, int stride) {
	auto result = ((const ifloat4s256*)varyings)[1];
	auto& co = ((ifloat4s256*)colorOutput)[0];
	auto dco = ifritSampleTex(0, result.x, 1.0 - result.y);
	co.x = dco.x;
	co.y = dco.y;
	co.z = dco.z;
	co.w = dco.w;
}
IFRIT_HOST Ifrit::Engine::FragmentShader* DemoFragmentShaderCuda::getCudaClone() {
	return Ifrit::Core::CUDA::hostGetDeviceObjectCopy<DemoFragmentShaderCuda>(this);
}