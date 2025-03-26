
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
namespace Ifrit::Graphics::SoftGraphics
{

	class ShaderRuntime
	{
	public:
		virtual ~ShaderRuntime() = default;
		virtual void						   loadIR(std::string shaderCode, std::string shaderIdentifier) = 0;
		virtual void*						   lookupSymbol(std::string symbol) = 0;
		virtual std::unique_ptr<ShaderRuntime> getThreadLocalCopy() = 0;
	};

	class ShaderRuntimeBuilder
	{
	public:
		virtual std::unique_ptr<ShaderRuntime> buildRuntime() const = 0;
	};
} // namespace Ifrit::Graphics::SoftGraphics