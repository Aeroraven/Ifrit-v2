
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

#include "ifrit/softgraphics/engine/export/ComLLVMRtExport.h"
#include "ifrit/softgraphics/engine/comllvmrt/WrappedLLVMRuntime.h"

IFRIT_APIDECL_COMPAT void* IFRIT_APICALL ifvmCreateLLVMRuntimeBuilder()
	IFRIT_EXPORT_COMPAT_NOTHROW
{
	return new Ifrit::Graphics::SoftGraphics::ComLLVMRuntime::
		WrappedLLVMRuntimeBuilder();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifvmDestroyLLVMRuntimeBuilder(void* p)
	IFRIT_EXPORT_COMPAT_NOTHROW
{
	delete (Ifrit::Graphics::SoftGraphics::ComLLVMRuntime::
			WrappedLLVMRuntimeBuilder*)p;
}