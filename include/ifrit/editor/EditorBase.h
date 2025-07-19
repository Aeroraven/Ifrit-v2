#pragma once
#include "ifrit/core/platform/ApiConv.h"
#ifndef IFRIT_MODULE_EDITOR
    #define IFRIT_EDITOR_API IFRIT_APIDECL_IMPORT
#else
    #define IFRIT_EDITOR_API IFRIT_APIDECL
#endif
