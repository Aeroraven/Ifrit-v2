#include "engine/shadervm/spirv/SpvVMExternalFunction.h"
extern "C" {
	IFRIT_APIDECL_FORCED void ifritShaderOps_Base_ImageWrite_v2i32_v4f32(
		void* pImage,
		ifritShaderOps_Base_Veci2 coord,
		ifritShaderOps_Base_Vecf4 color
	) {
		using namespace Ifrit::Core::Data;
		auto image = reinterpret_cast<ImageF32*>(pImage);
		image->fillPixelRGBA(coord.x, coord.y, color.x, color.y, color.z, color.w);
	}
}
