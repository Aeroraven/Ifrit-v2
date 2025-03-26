
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
#include "ifrit/softgraphics/engine/base/FrameBuffer.h"
#include "ifrit/softgraphics/engine/base/VaryingDescriptor.h"

#define IFRIT_BASENS Ifrit::Graphics::SoftGraphics
#define IFRIT_CORENS Ifrit::Graphics::SoftGraphics::Core::Data

namespace Ifrit::Graphics::SoftGraphics::LibraryExport
{
	using ExportTypeDesc = int;
}

/* Exporting FrameBuffer.h */
IFRIT_APIDECL_COMPAT					IFRIT_BASENS::FrameBuffer* IFRIT_APICALL
										iftrCreateFrameBuffer() IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyFrameBuffer(
	IFRIT_BASENS::FrameBuffer* pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetFrameBufferColorAttachmentFP32(
	IFRIT_BASENS::FrameBuffer* pInstance, IFRIT_CORENS::ImageF32* const* pImage,
	size_t nums) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetFrameBufferDepthAttachmentFP32(
	IFRIT_BASENS::FrameBuffer* pInstance,
	IFRIT_CORENS::ImageF32*	   pImage) IFRIT_EXPORT_COMPAT_NOTHROW;

/* Exporting VaryingDescriptors.h */
IFRIT_APIDECL_COMPAT					IFRIT_BASENS::VaryingDescriptor* IFRIT_APICALL
										iftrCreateVaryingDescriptor() IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyVaryingDescriptor(
	IFRIT_BASENS::VaryingDescriptor* pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrWriteVaryingDescriptor(
	IFRIT_BASENS::VaryingDescriptor*									pInstance,
	const Ifrit::Graphics::SoftGraphics::LibraryExport::ExportTypeDesc* pDesc,
	size_t																num) IFRIT_EXPORT_COMPAT_NOTHROW;

/* Exporting VertexBuffer.h */
IFRIT_APIDECL_COMPAT					IFRIT_BASENS::VertexBuffer* IFRIT_APICALL
										iftrCreateVertexBuffer() IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrDestroyVertexBuffer(
	IFRIT_BASENS::VertexBuffer* pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetVertexBufferLayout(
	IFRIT_BASENS::VertexBuffer*											pInstance,
	const Ifrit::Graphics::SoftGraphics::LibraryExport::ExportTypeDesc* pDesc,
	size_t																num) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
										iftrSetVertexBufferSize(IFRIT_BASENS::VertexBuffer* pInstance,
											size_t											num) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrAllocateVertexBuffer(
	IFRIT_BASENS::VertexBuffer* pInstance) IFRIT_EXPORT_COMPAT_NOTHROW;
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetVertexBufferValueFloat4(
	IFRIT_BASENS::VertexBuffer* pInstance, int index, int attr,
	void* value) IFRIT_EXPORT_COMPAT_NOTHROW;

#undef IFRIT_CORENS
#undef IFRIT_BASENS