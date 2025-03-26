
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

#include "ifrit/softgraphics/core/export/CommonExport.h"

#define IFRIT_CORENS Ifrit::Graphics::SoftGraphics::Core::Data
using namespace Ifrit::Graphics::SoftGraphics::Core::Data;

IFRIT_APIDECL_COMPAT IFRIT_CORENS::ImageF32* IFRIT_APICALL ifcrCreateImageFP32(
	size_t width, size_t height, size_t channel) IFRIT_EXPORT_COMPAT_NOTHROW
{
	return new ImageF32(width, height, channel);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifcrDestroyImageFP32(
	IFRIT_CORENS::ImageF32* pInstance) IFRIT_EXPORT_COMPAT_NOTHROW
{
	delete pInstance;
}
IFRIT_APIDECL_COMPAT float* IFRIT_APICALL ifcrGetImageRawDataFP32(
	IFRIT_CORENS::ImageF32* pInstance) IFRIT_EXPORT_COMPAT_NOTHROW
{
	return pInstance->getData();
}

#undef IFRIT_CORENS