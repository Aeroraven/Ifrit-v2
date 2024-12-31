
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
#include "ifrit/softgraphics/core/data/Image.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"

#define IFRIT_CORENS Ifrit::GraphicsBackend::SoftGraphics::Core::Data

/* Export Image.h */
IFRIT_APIDECL_COMPAT IFRIT_CORENS::ImageF32 *IFRIT_APICALL ifcrCreateImageFP32(
    size_t width, size_t height, size_t channel) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifcrDestroyImageFP32(
    IFRIT_CORENS::ImageF32 *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT float *IFRIT_APICALL ifcrGetImageRawDataFP32(
    IFRIT_CORENS::ImageF32 *pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;

#undef IFRIT_CORENS