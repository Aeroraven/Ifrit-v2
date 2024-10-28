#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMExternalFunction.h"
#include "ifrit/softgraphics/engine/imaging/BufferedImageSampler.h"
extern "C" {
IFRIT_APIDECL_FORCED void
ifritShaderOps_Base_ImageWrite_v2i32_v4f32(void *pImage,
                                           ifritShaderOps_Base_Veci2 coord,
                                           ifritShaderOps_Base_Vecf4 color) {
  using namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Core::Data;
  auto image = reinterpret_cast<ImageF32 *>(pImage);
  image->fillPixelRGBA(coord.x, coord.y, color.x, color.y, color.z, color.w);
}

IFRIT_APIDECL_FORCED void ifritShaderOps_Base_ImageSampleExplicitLod_2d_v4f32(
    void *pSampledImage, ifritShaderOps_Base_Veci2 coord, float lod,
    ifritShaderOps_Base_Vecf4 *result) {
  auto pSi = (Ifrit::Engine::GraphicsBackend::SoftGraphics::Imaging::
                  BufferedImageSampler *)(pSampledImage);
  pSi->sample2DLodSi(coord.x, coord.y, lod, {0, 0}, result);
}
}
