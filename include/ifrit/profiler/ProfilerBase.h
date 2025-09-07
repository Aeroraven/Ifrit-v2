#pragma once
#include "ifrit/core/platform/ApiConv.h"
#ifndef IFRIT_MODULE_PROFILER
    #define IFRIT_PROFILER_API IFRIT_APIDECL_IMPORT
#else
    #define IFRIT_PROFILER_API IFRIT_APIDECL
#endif
