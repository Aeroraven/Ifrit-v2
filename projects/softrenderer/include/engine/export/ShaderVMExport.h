#pragma once
#include "./core/definition/CoreExports.h"

//Update v1
IFRIT_APIDECL_COMPAT void* IFRIT_APICALL ifspvmCreateVertexShaderFromFile(void* runtime, const char* path) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifspvmDestroyVertexShaderFromFile(void* p) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void* IFRIT_APICALL ifspvmFragmentShaderFromFile(void* runtime, const char* path) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifspvmDestroyFragmentShaderFromFile(void* p) IFRIT_EXPORT_COMPAT_NOTHROW;