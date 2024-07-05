#pragma once
#include "./core/definition/CoreTypes.h"
#include "./core/definition/CoreDefs.h"
#include "engine/base/Structures.h"
#include "engine/math/ShaderOpsCuda.cuh"

namespace Ifrit::Engine::Math::ShaderOps::CUDA {
	IFRIT_DUAL inline float4 textureImpl(const IfritSamplerT& sampler, const float4* texData,
		const int& texW, const int& texH, const float2& uv, const int2& offset, const float lod) {
		float cX = uv.x * (texW-1) + offset.x;
		float cY = uv.y * (texH-1) + offset.y;
		int pX, pY;
		float4 borderColor = ((sampler.borderColor == IF_BORDER_COLOR_BLACK) ? float4{ 0.0f,0.0f,0.0f,1.0f }: float4{1.0f, 1.0f, 1.0f, 1.0f});
		//Address Mode U
		if (sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_REPEAT) {
			if (cX < 0)cX += texW;
			cX = fmodf(cX, texW);
		}
		else if(sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) {
			cX = clamp(cX, 0.0f, texW - 1.0f);
		}
		else if (sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) {
			if (cX >= texW || cX < 0.0) {
				return borderColor;
			}
		}
		//Address Mode V
		if (sampler.addressModeV == IF_SAMPLER_ADDRESS_MODE_REPEAT) {
			if (cY < 0)cY += texH;
			cY = fmodf(cY, texH);
		}
		else if (sampler.addressModeV == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) {
			cY = clamp(cY, 0.0f, texH - 1.0f);
		}
		else if (sampler.addressModeV == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) {
			if (cY >= texH || cY < 0.0) {
				return borderColor;
			}
		}
		//Filter Mode
		if (sampler.filterMode == IF_FILTER_NEAREST) {
			pX = round(cX);
			pY = round(cY);
			pX %= texW;
			pY %= texH;
			
			return texData[pY * texW + pX];
		}
		else if (sampler.filterMode == IF_FILTER_LINEAR) {
			pX = (int)cX;
			pY = (int)cY;
			float propX = cX - (int)pX;
			float propY = cY - (int)pY;
			float4 c00 = texData[pY * texW + pX];
			float4 c01 = texData[pY * texW + min(pX + 1, texW - 1)];
			float4 c10 = texData[min(pY + 1, texH - 1) * texW + pX];
			float4 c11 = texData[min(pY + 1, texH - 1) * texW + min(pX + 1, texW - 1)];
			using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;
			float4 c0x = lerp(c00, c01, propX);
			float4 c1x = lerp(c10, c11, propX);
			return lerp(c0x, c1x, propY);
		}
	}
}

#define isbcuSampleTex(sampler,texture,uv) (Ifrit::Engine::Math::ShaderOps::CUDA::textureImpl(atSamplerPtr[(sampler)],(float4*)(atTexture[(texture)]),atTextureWid[(texture)],atTextureHei[(texture)],(uv),{0,0},0))
#define isbcuReadPsVarying(x,y)  ((const ifloat4s256*)(x))[(y)]
#define isbcuReadPsColorOut(x,y)  ((ifloat4s256*)(x))[(y)]
