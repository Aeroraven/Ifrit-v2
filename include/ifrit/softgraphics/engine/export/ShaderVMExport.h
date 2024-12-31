
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"

// Update v1
IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifspvmCreateVertexShaderFromFile(
    void *runtime, const char *path) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
ifspvmDestroyVertexShaderFromFile(void *p) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifspvmFragmentShaderFromFile(
    void *runtime, const char *path) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
ifspvmDestroyFragmentShaderFromFile(void *p) IFRIT_EXPORT_COMPAT_NOTHROW;