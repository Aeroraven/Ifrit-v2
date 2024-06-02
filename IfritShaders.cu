#include "IfritShaders.cuh"
#include "core/cuda/CudaUtils.cuh"

IFRIT_DUAL void DemoVertexShaderCuda::execute(const void* const* input, ifloat4* outPos, Ifrit::Engine::VaryingStore** outVaryings) {
	using namespace Ifrit::Engine::Math::ShaderOps::CUDA;
	//float4x4 view = (lookAt({ 0,1.5,5.25 }, { 0,1.5,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 0,0.75,1.50 }, { 0,0.75,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 0,0.1,1.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 0,0.0,2.25 }, { 0,0.0,0.0 }, { 0,1,0 }));
	float4x4 view = (lookAt({ 500,300,0 }, { -100,300,-0 }, { 0,1,0 }));
	float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 10.0, 3000));
	float4x4 mvp = multiply(proj, view);
	auto s = *reinterpret_cast<const ifloat4*>(input[0]);
	auto p = multiply(mvp, s);
	*outPos = p;

	outVaryings[0]->vf4 = *reinterpret_cast<const ifloat4*>(input[1]);
}
IFRIT_HOST Ifrit::Engine::VertexShader* DemoVertexShaderCuda::getCudaClone() {
	return Ifrit::Core::CUDA::hostGetDeviceObjectCopy<DemoVertexShaderCuda>(this);
}

IFRIT_DUAL void DemoFragmentShaderCuda::execute(const  Ifrit::Engine::VaryingStore* varyings, ifloat4* colorOutput) {
	ifloat4 result = varyings[0].vf4;
	constexpr float fw = 0.5;
	result.x = fw * result.x + fw;
	result.y = fw * result.y + fw;
	result.z = fw * result.z + fw;
	result.w = fw * result.w + fw;
	colorOutput[0] = result;
}
IFRIT_HOST Ifrit::Engine::FragmentShader* DemoFragmentShaderCuda::getCudaClone() {
	return Ifrit::Core::CUDA::hostGetDeviceObjectCopy<DemoFragmentShaderCuda>(this);
}