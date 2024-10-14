#pragma once

#include "./core/definition/CoreExports.h"

// Update v2.0
IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifbufCreateBufferManager()
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifbufDestroyBufferManager(void *p)
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void *IFRIT_APICALL
ifbufCreateBuffer(void *pManager, size_t bufSize) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifbufDestroyBuffer(void *p)
    IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
ifbufBufferData(void *pBuffer, const void *pData, size_t offset,
                size_t size) IFRIT_EXPORT_COMPAT_NOTHROW;
