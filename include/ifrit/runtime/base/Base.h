#pragma once
#include "ifrit/core/platform/ApiConv.h"
#ifndef IFRIT_MODULE_RUNTIME
    #define IFRIT_RUNTIME_API IFRIT_APIDECL_IMPORT
#else
    #define IFRIT_RUNTIME_API IFRIT_APIDECL
#endif