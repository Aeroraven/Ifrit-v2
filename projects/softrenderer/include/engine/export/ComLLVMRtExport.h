#pragma once

#include "./core/definition/CoreExports.h"

#define IFRIT_TRNS Ifrit::Engine::SoftRenderer::TileRaster
#define IFRIT_BASENS Ifrit::Engine::SoftRenderer

// Update v1
IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifvmCreateLLVMRuntimeBuilder()
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifvmDestroyLLVMRuntimeBuilder(void *p)
    IFRIT_EXPORT_COMPAT_NOTHROW;

#undef IFRIT_BASENS
#undef IFRIT_TRNS