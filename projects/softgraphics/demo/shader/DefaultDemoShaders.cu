
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#ifdef IFRIT_FEATURE_CUDA
#include "DefaultDemoShaders.cuh"
#include "core/cuda/CudaUtils.cuh"
#include "engine/math/ShaderBuiltinCuda.cuh"

namespace Ifrit::Demo::DemoDefault {
	IFRIT_DUAL void DemoVertexShaderCuda::execute(const void* const* input, ifloat4* outPos, ifloat4* const* outVaryings) {
		using namespace Ifrit::SoftRenderer::Math::ShaderOps::CUDA;
		//float4x4 view = (lookAt({ 0,1.5,5.25 }, { 0,1.5,0.0 }, { 0,1,0 }));
		//float4x4 view = (lookAt({ 0,0.75,1.50 }, { 0,0.75,0.0 }, { 0,1,0 }));
		//float4x4 view = (lookAt({ 0,0.1,1.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
		float4x4 view = (lookAt({ 0.08,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));  //fox
		//float4x4 view = (lookAt({ 0.0,0.6,-1.5 }, { 0,0.4,0.0 }, { 0,1,0 }));  //af 
		 
		//float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
		//float4x4 view = (lookAt({ 500,300,0 }, { -100,300,-0 }, { 0,1,0 }));
		//float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 10.0, 3000));

		//float4x4 view = (lookAt({ 0,1.5,0}, { -100,1.5,0 }, { 0,1,0 }));
		float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.1, 1000));
		float4x4 mvp = multiply(proj, view);
		auto s = isbReadFloat4(input[0]);
		auto p = multiply(mvp, s);
		*outPos = p;
		*outVaryings[0] = isbReadFloat4(input[1]);
		*outVaryings[1] = isbReadFloat4(input[2]);
		(*outVaryings[1]).y = 1.0f - outVaryings[1]->y;
	}

	IFRIT_HOST Ifrit::SoftRenderer::VertexShader* DemoVertexShaderCuda::getCudaClone() {
		return Ifrit::SoftRenderer::Core::CUDA::hostGetDeviceObjectCopy<DemoVertexShaderCuda>(this);
	}

	IFRIT_DUAL void DemoFragmentShaderCuda::execute(const  void* varyings, void* colorOutput, float* fragmentDepth) {
		using Ifrit::SoftRenderer::Math::ShaderOps::CUDA::abs;
		using Ifrit::SoftRenderer::Math::ShaderOps::CUDA::texture;
		using Ifrit::SoftRenderer::Math::ShaderOps::CUDA::textureLod;

		auto result = isbcuReadPsVarying(varyings, 0);
		auto& co = isbcuReadPsColorOut(colorOutput, 0);
		//auto dco = isbcuSampleTexLod(0, 0, float2( result.x, 1.0f - result.y ),2.5f); 
		//auto dcl = static_cast<const ifloat4s256*>(varyings);
		//float2 uv = { dcl[1].x,dcl[1].y };
		//auto dco = texture(0, 0, dcl, 1);

		co.x = result.x * 0.5 + 0.5;
		co.y = result.y * 0.5 + 0.5;
		co.z = result.z * 0.5 + 0.5;
		co.w = 0.5;

		/*
		co.x = result.x;
		co.y = result.y;
		co.z = result.z;
		co.w = 0.5;*/
		//printf("%f %f %f %f\n", result.x, result.y, result.z, result.w);
	}

	IFRIT_HOST Ifrit::SoftRenderer::FragmentShader* DemoFragmentShaderCuda::getCudaClone() {
		return Ifrit::SoftRenderer::Core::CUDA::hostGetDeviceObjectCopy<DemoFragmentShaderCuda>(this);
	}

	IFRIT_DUAL void DemoGeometryShaderCuda::execute(const ifloat4* const* inPos, const Ifrit::SoftRenderer::VaryingStore* const* inVaryings,
		ifloat4* outPos, Ifrit::SoftRenderer::VaryingStore* outVaryings, int* outSize) {
		outPos[0] = *inPos[0];
		outPos[1] = *inPos[1];
		outPos[2] = *inPos[2];

		outPos[0].x += 0.03;
		outPos[1].x += 0.03;
		outPos[2].x += 0.03;

		isbStoreGsVarying(0, 0, 2, isbReadGsVarying(0, 0));
		isbStoreGsVarying(0, 1, 2, isbReadGsVarying(0, 1));
		isbStoreGsVarying(1, 0, 2, isbReadGsVarying(1, 0));
		isbStoreGsVarying(1, 1, 2, isbReadGsVarying(1, 1));
		isbStoreGsVarying(2, 0, 2, isbReadGsVarying(2, 0));
		isbStoreGsVarying(2, 1, 2, isbReadGsVarying(2, 1));
		*outSize = 3;
	}

	IFRIT_HOST Ifrit::SoftRenderer::GeometryShader* DemoGeometryShaderCuda::getCudaClone() {
		return  Ifrit::SoftRenderer::Core::CUDA::hostGetDeviceObjectCopy<DemoGeometryShaderCuda>(this);
	}
}
#endif