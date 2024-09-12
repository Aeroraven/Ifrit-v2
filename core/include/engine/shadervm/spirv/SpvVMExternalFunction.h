#pragma once
#include "core/data/Image.h"


extern "C" {
	struct IFRIT_APIDECL_FORCED  alignas(16) ifritShaderOps_Base_Vecf4 {
		float x, y, z, w;
	};
	struct IFRIT_APIDECL_FORCED  alignas(16) ifritShaderOps_Base_Veci2 {
		int x, y;
	};

	IFRIT_APIDECL_FORCED void ifritShaderOps_Base_ImageWrite_v2i32_v4f32(
		void* pImage,
		ifritShaderOps_Base_Veci2 coord,
		ifritShaderOps_Base_Vecf4 color
	);
}