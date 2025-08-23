#pragma once

#include "ifrit/core/platform/ApiConv.h"

#ifdef __INTELLISENSE__
    #define IFRIT_RHI_API
#else
    #ifdef IFRIT_MODULE_RHI
        #define IFRIT_RHI_API IFRIT_API_EXPORT
    #else
        #define IFRIT_RHI_API IFRIT_API_IMPORT
    #endif
#endif