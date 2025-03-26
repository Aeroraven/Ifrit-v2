
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
namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv
{
	enum SpvVMExtRegistryTypeIdentifier
	{
		IFSP_EXTREG_TP_INT,
		IFSP_EXTREG_TP_FLOAT
	};

	typedef std::string (*SpvVMExtRegistryFunctionGenerator)(
		const std::vector<SpvVMExtRegistryTypeIdentifier>&,
		const std::vector<int>&);

	class SpvVMExtRegistry
	{
	private:
		std::unordered_map<std::string,
			std::unordered_map<int, SpvVMExtRegistryFunctionGenerator>>
										generators;
		std::unordered_set<std::string> registeredFunc;
		std::string						irCode;

	public:
		SpvVMExtRegistry();
		std::string queryExternalFunc(
			std::string extImportName, int functionName,
			const std::vector<SpvVMExtRegistryTypeIdentifier>& identifiers,
			const std::vector<int>&							   componentSize);
		std::string getRequiredFuncDefs();
	};
} // namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv