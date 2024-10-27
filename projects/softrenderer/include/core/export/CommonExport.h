#pragma once
#include "./core/data/Image.h"
#include "./core/definition/CoreExports.h"

#define IFRIT_CORENS Ifrit::Engine::GraphicsBackend::SoftGraphics::Core::Data

/* Export Image.h */
IFRIT_APIDECL_COMPAT IFRIT_CORENS::ImageF32 *IFRIT_APICALL ifcrCreateImageFP32(
    size_t width, size_t height, size_t channel) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifcrDestroyImageFP32(
    IFRIT_CORENS::ImageF32 *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT float *IFRIT_APICALL ifcrGetImageRawDataFP32(
    IFRIT_CORENS::ImageF32 *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;

#undef IFRIT_CORENS