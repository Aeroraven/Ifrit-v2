#pragma once

#include "ifrit/core/platform/ApiConv.h"

#ifdef __INTELLISENSE__
    #define IFRIT_VKRHI2_API
#else
    #ifdef IFRIT_MODULE_VKRHI2
        #define IFRIT_VKRHI2_API IFRIT_APIDECL
    #else
        #define IFRIT_VKRHI2_API IFRIT_APIDECL_IMPORT
    #endif
#endif