
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


#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterConstantsCuda.h"
namespace Ifrit::SoftRenderer::TileRaster::CUDA::Invocation::Impl {

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