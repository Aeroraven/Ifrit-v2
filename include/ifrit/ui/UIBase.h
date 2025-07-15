#pragma once
#include "ifrit/core/platform/ApiConv.h"
#ifndef IFRIT_MODULE_UI
    #define IFRIT_UI_API IFRIT_APIDECL_IMPORT
#else
    #define IFRIT_UI_API IFRIT_APIDECL
#endif