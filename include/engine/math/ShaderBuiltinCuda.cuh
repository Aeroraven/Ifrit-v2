#pragma once
#include "./core/definition/CoreTypes.h"
#include "./core/definition/CoreDefs.h"
#include "engine/base/Structures.h"
#include "engine/math/ShaderOpsCuda.cuh"

#include "./engine/tilerastercuda/TileRasterCommonResourceCuda.cuh"

#ifdef IFRIT_FEATURE_CUDA
namespace Ifrit::Engine::Math::ShaderOps::CUDA {
	using TextureObject = int;
	using SamplerState = int;
	using DifferentiableCollection = const ifloat4s256*;
	using DifferentiableVarId = int;

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

	namespace TextureSampleImpl {
		IFRIT_DUAL inline int textureLodPtroffImpl(int texW, int texH, int texLayers, int lod) {
			int off = 0, w = texW, h = texH;
			for (int i = 0; i < lod; i++) {
				off += w * h * texLayers;
				w = (w + 1) >> 1;
				h = (h + 1) >> 1;
			}
			return off;
		}
		IFRIT_DUAL inline float4 textureImplLegacy(const IfritSamplerT& sampler, const float4* texData,
			int texOw, int texOh, int texLayers, const float2& uv, const int2& offset, int lod) {
			
			int lodoff = textureLodPtroffImpl(texOw, texOh, texLayers, lod);
			int texW = IFRIT_InvoCeilRshift(texOw, lod);
			int texH = IFRIT_InvoCeilRshift(texOh, lod);
			float cX = uv.x * (texW - 1) + offset.x;
			float cY = uv.y * (texH - 1) + offset.y;
			int pX, pY;
			float4 borderColor = ((sampler.borderColor == IF_BORDER_COLOR_BLACK) ? float4{ 0.0f,0.0f,0.0f,1.0f } : float4{ 1.0f, 1.0f, 1.0f, 1.0f });
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
			else if (sampler.addressModeU == IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) {
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

		IFRIT_DUAL inline float4 textureImplLod(const IfritSamplerT& sampler, const float4* texData,
			int texOw, int texOh, int texLayers, const float2& uv, const int2& offset, const float lod) {
			
			int lodLow = floor(lod);
			auto lodDiff = lod - lodLow;
			auto texcLow = textureImplLegacy(sampler, texData, texOw, texOh, texLayers, uv, offset, lodLow);
			auto texcHigh = textureImplLegacy(sampler, texData, texOw, texOh, texLayers, uv, offset, lodLow + 1);
			return lerp(texcLow, texcHigh, lodDiff);
		}

		IFRIT_DUAL inline float4 textureImplAdaptiveLod(const IfritSamplerT& sampler, const float4* texData,
			int texOw, int texOh,int texLayers, const float2& uv, const float2& dx, const float2& dy, float lodBias, const int2& offset) {
			
			const float2 dxI = { dx.x * texOw,dx.y * texOh };
			const float2 dyI = { dy.x * texOw,dx.y * texOh };
			const auto dxI2 = dxI.x * dxI.x + dxI.y * dxI.y;
			const auto dyI2 = dyI.x * dyI.x + dyI.y * dyI.y;
			const auto d = max(dxI2, dyI2);
			const auto lod = max(0.0f, 0.5f * log2(d) + lodBias);
			return textureImplLod(sampler, texData, texOw, texOh, texLayers, uv, offset, lod);
		}
		IFRIT_DUAL inline float4 textureImplAnisotropicFilter(const IfritSamplerT& sampler, const float4* texData,
			int texOw, int texOh, int texLayers, int maxAniso, const float2& uv, const float2& du, const float2& dv, float lodBias, const int2& offset) {
			//TODO: Pixel size
			auto dudx = du.x, dudy = du.y, dvdx = dv.x, dvdy = dv.y;
			auto px = sqrt(dudx * dudx + dvdx * dvdx) * texOw;
			auto py = sqrt(dudy * dudy + dvdy * dvdy) * texOh;
			auto pmax = max(px, py), pmin = min(px, py);
			int n = max(1.0f, min(ceil(pmax / pmin) + 1e-3, 1.0f * maxAniso + 1e-3));
			auto lambda = max(0.0f,log2(pmax / n));
			auto ddu = (px > py) ? dudx : dudy;
			auto ddv = (px > py) ? dvdx : dvdy;
			float4 sumAniso = { 0.0f,0.0f,0.0f,0.0f };
			for (int i = 0; i < n; i++) {
				float2 uv2;
				uv2.x = uv.x + ddu * (1.0f * i / (n + 1) - 0.5f);
				uv2.y = uv.y + ddv * (1.0f * i / (n + 1) - 0.5f);
				auto res = textureImplLod(sampler, texData, texOw, texOh, texLayers, uv2, offset, lambda + lodBias);
				sumAniso.x += res.x;
				sumAniso.y += res.y;
				sumAniso.z += res.z;
				sumAniso.w += res.w;
			}
			sumAniso.x /= n;
			sumAniso.y /= n;
			sumAniso.z /= n;
			sumAniso.w /= n;
			return sumAniso;
		}

		IFRIT_DUAL inline float4 textureImplAdaptiveLodFromAttr(const IfritSamplerT& sampler, const float4* texData,
			int texOw, int texOh, int texLayers, const float lodBias, const int2& offset, const ifloat4s256* varyings, int varyId) {
			
			auto uvRaw = ((const ifloat4s256*)(varyings))[varyId];
			float2 uv = { uvRaw.x,uvRaw.y };
			auto dx = pixelDfDx_s256(varyings, varyId);
			auto dy = pixelDfDy_s256(varyings, varyId);
			float2 ndx = { dx.x, dx.y };
			float2 ndy = { dy.x, dy.y };
			return textureImplAdaptiveLod(sampler, texData, texOw, texOh, texLayers, uv, ndx, ndy, lodBias, offset);
		}
		IFRIT_DUAL inline float4 textureImplAnisotropicFilterFromAttr(const IfritSamplerT& sampler, const float4* texData,
			int texOw, int texOh, int texLayers, float maxAniso, float lodBias, const int2& offset, const ifloat4s256* varyings, int varyId) {
			auto uvRaw = ((const ifloat4s256*)(varyings))[varyId];
			float2 uv = { uvRaw.x,uvRaw.y };
			auto dx = pixelDfDx_s256(varyings, varyId);
			auto dy = pixelDfDy_s256(varyings, varyId);
			float2 ndx = { dx.x, dx.y };
			float2 ndy = { dy.x, dy.y };
			return textureImplAnisotropicFilter(sampler, texData, texOw, texOh, texLayers, maxAniso, uv, ndx, ndy, lodBias, offset);
		}
	}
	
	IFRIT_DUAL inline float4 textureLod(TextureObject tex, SamplerState samplerState,const float2& uv,float lod) {
#ifndef __CUDA_ARCH__
		printf("This function is not available under CPU Mode.");
		return { 0,0,0,0 };
#else
		using namespace Ifrit::Engine::TileRaster::CUDA::Invocation;
		auto& pSamplerState = Impl::csSamplers[samplerState];
		float4* pTexture = reinterpret_cast<float4*>(Impl::csTextures[tex]);
		auto pHeight = Impl::csTextureHeight[tex];
		auto pWidth = Impl::csTextureWidth[tex];
		auto pLayers = Impl::csTextureArrayLayers[tex];
		return TextureSampleImpl::textureImplLod(pSamplerState, pTexture, pWidth, pHeight, pLayers, uv, { 0,0 }, lod);
#endif
	}
	
	IFRIT_DUAL inline float4 textureLodOffset(TextureObject tex, SamplerState samplerState, const float2& uv, float lod,const int2& offset) {
#ifndef __CUDA_ARCH__
		printf("This function is not available under CPU Mode.");
		return { 0,0,0,0 };
#else
		using namespace Ifrit::Engine::TileRaster::CUDA::Invocation;
		auto& pSamplerState = Impl::csSamplers[samplerState];
		float4* pTexture = reinterpret_cast<float4*>(Impl::csTextures[tex]);
		auto pHeight = Impl::csTextureHeight[tex];
		auto pWidth = Impl::csTextureWidth[tex];
		auto pLayers = Impl::csTextureArrayLayers[tex];
		return TextureSampleImpl::textureImplLod(pSamplerState, pTexture, pWidth, pHeight,pLayers, uv, offset, lod);
#endif
	}
	
	IFRIT_DUAL inline float4 textureGrad(TextureObject tex, SamplerState samplerState, const float2& uv, const float2& dPdx,const float2& dPdy) {
#ifndef __CUDA_ARCH__
		printf("This function is not available under CPU Mode.");
		return { 0,0,0,0 };
#else
		using namespace Ifrit::Engine::TileRaster::CUDA::Invocation;
		auto& pSamplerState = Impl::csSamplers[samplerState];
		float4* pTexture = reinterpret_cast<float4*>(Impl::csTextures[tex]);
		auto pHeight = Impl::csTextureHeight[tex];
		auto pWidth = Impl::csTextureWidth[tex];
		auto pLayers = Impl::csTextureArrayLayers[tex];
		return TextureSampleImpl::textureImplAdaptiveLod(pSamplerState, pTexture, pWidth, pHeight,pLayers, uv, dPdx, dPdy, 0.0f, { 0,0 });
#endif
	}

	IFRIT_DUAL inline float4 textureGradOffset(TextureObject tex, SamplerState samplerState, const float2& uv, const float2& dPdx, const float2& dPdy, const int2& offset) {
#ifndef __CUDA_ARCH__
		printf("This function is not available under CPU Mode.");
		return { 0,0,0,0 };
#else
		using namespace Ifrit::Engine::TileRaster::CUDA::Invocation;
		auto& pSamplerState = Impl::csSamplers[samplerState];
		float4* pTexture = reinterpret_cast<float4*>(Impl::csTextures[tex]);
		auto pHeight = Impl::csTextureHeight[tex];
		auto pWidth = Impl::csTextureWidth[tex];
		auto pLayers = Impl::csTextureArrayLayers[tex];
		return TextureSampleImpl::textureImplAdaptiveLod(pSamplerState, pTexture, pWidth, pHeight, pLayers, uv, dPdx, dPdy, 0.0f, offset);
#endif
	}

	IFRIT_DUAL inline float4 texture(TextureObject tex, SamplerState samplerState, DifferentiableCollection var, DifferentiableVarId uv, float bias = 0.0f) {
#ifndef __CUDA_ARCH__
		printf("This function is not available under CPU Mode.");
		return { 0,0,0,0 };
#else
		using namespace Ifrit::Engine::TileRaster::CUDA::Invocation;
		auto& pSamplerState = Impl::csSamplers[samplerState];
		float4* pTexture = reinterpret_cast<float4*>(Impl::csTextures[tex]);
		auto pHeight = Impl::csTextureHeight[tex];
		auto pWidth = Impl::csTextureWidth[tex];
		auto pLayers = Impl::csTextureArrayLayers[tex];
		if (pSamplerState.anisotropyEnable) {
			return  TextureSampleImpl::textureImplAnisotropicFilterFromAttr(pSamplerState, pTexture, pWidth, pHeight,pLayers,
				pSamplerState.maxAnisotropy, bias, { 0,0 }, var, uv);
		}
		else {
			return TextureSampleImpl::textureImplAdaptiveLodFromAttr(pSamplerState, pTexture, pWidth, pHeight, pLayers,
				bias, { 0,0 }, var, uv);
		}
#endif
	}
}

//#define isbcuSampleTex(sampler,texture,uv) (Ifrit::Engine::Math::ShaderOps::CUDA::textureImpl(atSamplerPtr[(sampler)],(float4*)(atTexture[(texture)]),atTextureWid[(texture)],atTextureHei[(texture)],(uv),{0,0},0))
//#define isbcuSampleTexAttr(sampler,texture,vary,varyId,bias) (Ifrit::Engine::Math::ShaderOps::CUDA::textureImplAdaptiveLodFromAttr(atSamplerPtr[(sampler)],(float4*)(atTexture[(texture)]),atTextureWid[(texture)],atTextureHei[(texture)],(bias),{0,0},((const ifloat4s256*)(vary)),(varyId)))
//#define isbcuSampleTexAttr1(sampler,texture,vary,varyId,bias,offset) (Ifrit::Engine::Math::ShaderOps::CUDA::textureImplAdaptiveLodFromAttr(atSamplerPtr[(sampler)],(float4*)(atTexture[(texture)]),atTextureWid[(texture)],atTextureHei[(texture)],(bias),(offset),((const ifloat4s256*)(vary)),(varyId)))
//#define isbcuSampleTexLod(sampler,texture,uv,lod) (Ifrit::Engine::Math::ShaderOps::CUDA::textureImplLod(atSamplerPtr[(sampler)],(float4*)(atTexture[(texture)]),atTextureWid[(texture)],atTextureHei[(texture)],(uv),{0,0},(lod)))
#define isbcuReadPsVarying(x,y)  ((const ifloat4s256*)(x))[(y)]
#define isbcuReadPsColorOut(x,y)  ((ifloat4s256*)(x))[(y)]
#define isbcuDfDx(x,y)  (Ifrit::Engine::Math::ShaderOps::CUDA::pixelDfDx_s256(((const ifloat4s256*)(x)),(y)))
#define isbcuDfDy(x,y)  (Ifrit::Engine::Math::ShaderOps::CUDA::pixelDfDy_s256(((const ifloat4s256*)(x)),(y)))

#endif