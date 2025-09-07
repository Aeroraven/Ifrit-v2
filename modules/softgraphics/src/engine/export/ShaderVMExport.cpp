
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

#include "ifrit/softgraphics/engine/export/ShaderVMExport.h"
#include "ifrit/softgraphics/engine/base/ShaderRuntime.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMInterpreter.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMReader.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMShader.h"

using namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv;
using namespace Ifrit::Graphics::SoftGraphics;

IFRIT_APIDECL_COMPAT void* IFRIT_APICALL ifspvmCreateVertexShaderFromFile(
	void* runtime, const char* path) IFRIT_EXPORT_COMPAT_NOTHROW
{
	SpvVMReader reader;
	auto		fsCode = reader.readFile(path);
	return new SpvVertexShader(*(ShaderRuntimeBuilder*)runtime, fsCode);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
ifspvmDestroyVertexShaderFromFile(void* p) IFRIT_EXPORT_COMPAT_NOTHROW
{
	delete (SpvVertexShader*)p;
}

IFRIT_APIDECL_COMPAT void* IFRIT_APICALL ifspvmCreateFragmentShaderFromFile(
	void* runtime, const char* path) IFRIT_EXPORT_COMPAT_NOTHROW
{
	SpvVMReader reader;
	auto		fsCode = reader.readFile(path);
	return new SpvFragmentShader(*(ShaderRuntimeBuilder*)runtime, fsCode);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
ifspvmDestroyFragmentShaderFromFile(void* p) IFRIT_EXPORT_COMPAT_NOTHROW
{
	delete (SpvFragmentShader*)p;
}