#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "core/definition/CoreExports.h"
#include "engine/tilerastercuda/TileRasterConstantsCuda.h"
namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {

	extern IFRIT_DEVICE_CONST float* csTextures[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST int csTextureWidth[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST int csTextureHeight[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST int csTextureMipLevels[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST int csTextureAnisotropicLevel[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST IfritSamplerT csSamplers[CU_MAX_SAMPLER_SLOTS];

	extern float* hsTextures[CU_MAX_TEXTURE_SLOTS];
	extern int hsTextureWidth[CU_MAX_TEXTURE_SLOTS];
	extern int hsTextureHeight[CU_MAX_TEXTURE_SLOTS];
	extern int hsTextureMipLevels[CU_MAX_TEXTURE_SLOTS];
	extern int hsTextureAnisotropicLevel[CU_MAX_TEXTURE_SLOTS];
	extern IfritSamplerT hsSampler[CU_MAX_SAMPLER_SLOTS];
}
#endif