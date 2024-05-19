#pragma once
#include "engine/base/VertexShader.h"
#include "engine/base/FragmentShader.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/tilerastercuda/TileRasterInvocationCuda.cuh"
class DemoVertexShaderCuda : public  Ifrit::Engine::VertexShader {

public:
	IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, Ifrit::Engine::VaryingStore** outVaryings) override {
		
		using namespace Ifrit::Engine::Math::ShaderOps::CUDA;
		float4x4 view = (lookAt({ 0,0.0,1.25 }, { 0,0.0,0.0 }, { 0,1,0 }));
		float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.1, 4000));
		float4x4 model;
		float4x4 mvp = multiply(proj, view);
		auto s = *reinterpret_cast<const ifloat4*>(input[0]);
		auto p = multiply(mvp, s);
		*outPos = p;
		outVaryings[0]->vf4 = *reinterpret_cast<const ifloat4*>(input[1]);

		printf("VS executed %f,%f,%f,%f\n", s.x, s.y, s.z, s.w);
	}
};

class DemoFragmentShaderCuda : public  Ifrit::Engine::FragmentShader {
public:
	IFRIT_DUAL virtual void execute(const  Ifrit::Engine::VaryingStore* varyings, ifloat4* colorOutput) override {
		ifloat4 result = varyings[0].vf4;
		result.x = 0.5 * result.x + 0.5;
		result.y = 0.5 * result.y + 0.5;
		result.z = 0.5 * result.z + 0.5;
		result.w = 0.5 * result.w + 0.5;

		colorOutput[0] = result;
		//printf("FS executed %4f,%4f,%4f,%4f\n", colorOutput[0].x, colorOutput[0].y, colorOutput[0].z, colorOutput[0].w);
	}
};

template DemoFragmentShaderCuda* Ifrit::Engine::TileRaster::CUDA::Invocation::copyShaderToDevice<DemoFragmentShaderCuda>(DemoFragmentShaderCuda* shader);
template DemoVertexShaderCuda* Ifrit::Engine::TileRaster::CUDA::Invocation::copyShaderToDevice<DemoVertexShaderCuda>(DemoVertexShaderCuda* shader);
