#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterConstantsCuda.h"
namespace Ifrit::Engine::SoftRenderer::TileRaster::CUDA::Invocation::Impl {

	extern IFRIT_DEVICE_CONST float* csTextures[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST int csTextureWidth[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST int csTextureHeight[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST int csTextureMipLevels[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST int csTextureArrayLayers[CU_MAX_TEXTURE_SLOTS];
	extern IFRIT_DEVICE_CONST IfritSamplerT csSamplers[CU_MAX_SAMPLER_SLOTS];
	extern IFRIT_DEVICE_CONST char* csGeneralBuffer[CU_MAX_BUFFER_SLOTS];
	extern IFRIT_DEVICE_CONST int csGeneralBufferSize[CU_MAX_BUFFER_SLOTS];

	extern float* hsTextures[CU_MAX_TEXTURE_SLOTS];
	extern int hsTextureWidth[CU_MAX_TEXTURE_SLOTS];
	extern int hsTextureHeight[CU_MAX_TEXTURE_SLOTS];
	extern int hsTextureMipLevels[CU_MAX_TEXTURE_SLOTS];
	extern int hsTextureArrayLayers[CU_MAX_TEXTURE_SLOTS];
	extern IfritSamplerT hsSampler[CU_MAX_SAMPLER_SLOTS];
	extern char* hsGeneralBuffer[CU_MAX_BUFFER_SLOTS];
	extern int hsGeneralBufferSize[CU_MAX_BUFFER_SLOTS];

}
#endif