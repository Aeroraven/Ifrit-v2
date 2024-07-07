#pragma once
#include "./core/definition/CoreTypes.h"
#include "./core/definition/CoreDefs.h"
#include "engine/base/Structures.h"
#include "engine/math/ShaderOpsCuda.cuh"

namespace Ifrit::Engine::Math::ShaderOps::CUDA {
	IFRIT_DUAL inline int textureLodPtroffImpl(const int& texW, const int& texH, const int& lod) {
		int off = 0, w = texW, h = texH;
		for (int i = 0; i < lod; i++) {
			off += w * h;
			w = (w + 1) >> 1;
			h = (h + 1) >> 1;
		}
		return off;
	}
	IFRIT_DUAL inline float4 textureImpl(const IfritSamplerT& sampler, const float4* texData,
		const int& texOw, const int& texOh, const float2& uv, const int2& offset, const int lod) {
		
		int lodoff = textureLodPtroffImpl(texOw, texOh, lod);
		int texW = IFRIT_InvoCeilRshift(texOw, lod);
		int texH = IFRIT_InvoCeilRshift(texOh, lod);
		float cX = uv.x * (texW - 1) + offset.x;
		float cY = uv.y * (texH - 1) + offset.y;
		int pX, pY;
		float4 borderColor = ((sampler.borderColor == IF_BORDER_COLOR_BLACK) ? float4{ 0.0f,0.0f,0.0f,1.0f }: float4{1.0f, 1.0f, 1.0f, 1.0f});
		//Address Mode U
		if (sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_REPEAT) {
			if (cX < 0)cX += texW;
			cX = fmodf(cX, texW);
		}
		else if (sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT) {
			if (cX < 0)cX += 2 * texW;
			cX = fmodf(cX, 2 * texW);
			cX = min(cX, 2 * texW - cX);
		}
		else if(sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) {
			cX = clamp(cX, 0.0f, texW - 1.0f);
		}
		else if (sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) {
			if (cX >= texW || cX < 0.0) {
				return borderColor;
			}
		}
		else if (sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE) {
			cX = mirrorclamp(cX, 0.0f, texW - 1.0f);
		}
		//Address Mode V
		if (sampler.addressModeV == IF_SAMPLER_ADDRESS_MODE_REPEAT) {
			if (cY < 0)cY += texH;
			cY = fmodf(cY, texH);
		}
		else if (sampler.addressModeV == IF_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT) {
			if (cY < 0)cY += 2 * texH;
			cY = fmodf(cY, 2 * texH);
			cY = min(cY, 2 * texH - cY);
		}
		else if (sampler.addressModeV == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) {
			cY = clamp(cY, 0.0f, texH - 1.0f);
		}
		else if (sampler.addressModeV == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) {
			if (cY >= texH || cY < 0.0) {
				return borderColor;
			}
		}
		else if (sampler.addressModeV == IF_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE) {
			cY = mirrorclamp(cY, 0.0f, texH - 1.0f);
		}
		//Filter Mode
		if (sampler.filterMode == IF_FILTER_NEAREST) {
			pX = round(cX);
			pY = round(cY);
			pX %= texW;
			pY %= texH;
			
			return texData[pY * texW + pX + lodoff];
		}
		else if (sampler.filterMode == IF_FILTER_LINEAR) {
			pX = (int)cX;
			pY = (int)cY;
			float propX = cX - (int)pX;
			float propY = cY - (int)pY;
			//printf("LODOFF %d\n", lodoff);
			float4 c00 = texData[pY * texW + pX + lodoff];
			float4 c01 = texData[pY * texW + min(pX + 1, texW - 1) + lodoff];
			float4 c10 = texData[min(pY + 1, texH - 1) * texW + pX + lodoff];
			float4 c11 = texData[min(pY + 1, texH - 1) * texW + min(pX + 1, texW - 1) + lodoff];
			using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;
			float4 c0x = lerp(c00, c01, propX);
			float4 c1x = lerp(c10, c11, propX);
			return lerp(c0x, c1x, propY);
		}
	}
	IFRIT_DEVICE inline float4 pixelDfDx_s256Device(const ifloat4s256* varyings, int varyId) {
		const float* ptr = reinterpret_cast<const float*>(varyings);
		const auto threadX = threadIdx.x;
		const auto internalX = threadX % 4;
		ptr = ptr - internalX;
		const auto offY = threadX & 2;
		ptr = ptr + offY;

		const auto ptrLeft = reinterpret_cast<const ifloat4s256*>(ptr);
		const auto ptrRight = reinterpret_cast<const ifloat4s256*>(ptr + 1);
		const auto leftObj = ptrLeft[varyId];
		const auto rightObj = ptrRight[varyId];

		float4 ret;
		ret.x = rightObj.x - leftObj.x;
		ret.y = rightObj.y - leftObj.y;
		ret.z = rightObj.z - leftObj.z;
		ret.w = rightObj.w - leftObj.w;
		return ret;
	}

	IFRIT_DEVICE inline float4 pixelDfDy_s256Device(const ifloat4s256* varyings, int varyId) {
		const float* ptr = reinterpret_cast<const float*>(varyings);
		const auto threadX = threadIdx.x;
		const auto internalX = threadX % 4;
		ptr = ptr - internalX;
		const auto offX = threadX & 1;
		ptr = ptr + offX;

		const auto ptrLeft = reinterpret_cast<const ifloat4s256*>(ptr);
		const auto ptrRight = reinterpret_cast<const ifloat4s256*>(ptr + 2);
		const auto leftObj = ptrLeft[varyId];
		const auto rightObj = ptrRight[varyId];

		float4 ret;
		ret.x = rightObj.x - leftObj.x;
		ret.y = rightObj.y - leftObj.y;
		ret.z = rightObj.z - leftObj.z;
		ret.w = rightObj.w - leftObj.w;
		return ret;
	}

	IFRIT_DUAL inline float4 pixelDfDx_s256(const ifloat4s256* varyings, int varyId) {
#ifdef __CUDA_ARCH__
		return pixelDfDx_s256Device(varyings, varyId);
#else
		printf("Shader derivatives ddx is not supported in CPU mode now\n");
		return float4{ 1.0f,0.0f,0.0f,0.0f };
#endif
	}

	IFRIT_DUAL inline float4 pixelDfDy_s256(const ifloat4s256* varyings, int varyId) {
#ifdef __CUDA_ARCH__
		return pixelDfDy_s256Device(varyings, varyId);
#else
		printf("Shader derivatives ddy is not supported in CPU mode now\n");
		return float4{ 1.0f,0.0f,0.0f,0.0f };
#endif
	}
}

#define isbcuSampleTex(sampler,texture,uv) (Ifrit::Engine::Math::ShaderOps::CUDA::textureImpl(atSamplerPtr[(sampler)],(float4*)(atTexture[(texture)]),atTextureWid[(texture)],atTextureHei[(texture)],(uv),{0,0},0))
#define isbcuSampleTexLod(sampler,texture,uv,lod) (Ifrit::Engine::Math::ShaderOps::CUDA::textureImpl(atSamplerPtr[(sampler)],(float4*)(atTexture[(texture)]),atTextureWid[(texture)],atTextureHei[(texture)],(uv),{0,0},(lod)))
#define isbcuReadPsVarying(x,y)  ((const ifloat4s256*)(x))[(y)]
#define isbcuReadPsColorOut(x,y)  ((ifloat4s256*)(x))[(y)]
#define isbcuDfDx(x,y)  (Ifrit::Engine::Math::ShaderOps::CUDA::pixelDfDx_s256(((const ifloat4s256*)(x)),(y)))
#define isbcuDfDy(x,y)  (Ifrit::Engine::Math::ShaderOps::CUDA::pixelDfDy_s256(((const ifloat4s256*)(x)),(y)))
