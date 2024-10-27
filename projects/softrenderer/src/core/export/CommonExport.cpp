#include "core/export/CommonExport.h"

#define IFRIT_CORENS Ifrit::Engine::GraphicsBackend::SoftGraphics::Core::Data
using namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Core::Data;

IFRIT_APIDECL_COMPAT IFRIT_CORENS::ImageF32 *IFRIT_APICALL ifcrCreateImageFP32(
    size_t width, size_t height, size_t channel) IFRIT_EXPORT_COMPAT_NOTHROW {
  return new ImageF32(width, height, channel);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifcrDestroyImageFP32(
    IFRIT_CORENS::ImageF32 *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
  delete pInstance;
}
IFRIT_APIDECL_COMPAT float *IFRIT_APICALL ifcrGetImageRawDataFP32(
    IFRIT_CORENS::ImageF32 *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
  return pInstance->getData();
}

#undef IFRIT_CORENS