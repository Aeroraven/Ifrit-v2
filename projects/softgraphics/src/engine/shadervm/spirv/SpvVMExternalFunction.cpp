
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

#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMExternalFunction.h"
#include "ifrit/softgraphics/engine/imaging/BufferedImageSampler.h"
extern "C"
{
	IFRIT_APIDECL_FORCED void
	ifritShaderOps_Base_ImageWrite_v2i32_v4f32(void* pImage,
		ifritShaderOps_Base_Veci2					 coord,
		ifritShaderOps_Base_Vecf4					 color)
	{
		using namespace Ifrit::Graphics::SoftGraphics::Core::Data;
		auto image = reinterpret_cast<ImageF32*>(pImage);
		image->fillPixelRGBA(coord.x, coord.y, color.x, color.y, color.z, color.w);
	}

	IFRIT_APIDECL_FORCED void ifritShaderOps_Base_ImageSampleExplicitLod_2d_v4f32(
		void* pSampledImage, ifritShaderOps_Base_Veci2 coord, float lod,
		ifritShaderOps_Base_Vecf4* result)
	{
		auto pSi =
			(Ifrit::Graphics::SoftGraphics::Imaging::BufferedImageSampler*)(pSampledImage);
		pSi->sample2DLodSi(coord.x * 1.0f, coord.y * 1.0f, lod, { 0, 0 }, result);
	}
}
