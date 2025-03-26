
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

#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMExtInstRegistry.h"
#include <spirv_headers/include/spirv/unified1/GLSL.std.450.h>

namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv
{
	std::string getLlvmType(SpvVMExtRegistryTypeIdentifier baseIdentifier,
		int												   paramComSize)
	{
		if (baseIdentifier == IFSP_EXTREG_TP_INT)
			return paramComSize == 1 ? "i32"
									 : ("<" + std::to_string(paramComSize) + " x i32>");
		if (baseIdentifier == IFSP_EXTREG_TP_FLOAT)
			return paramComSize == 1
				? "float"
				: ("<" + std::to_string(paramComSize) + " x float>");
		return "unknown";
	}
	std::string getMangledType(SpvVMExtRegistryTypeIdentifier baseIdentifier,
		int													  paramComSize)
	{
		if (baseIdentifier == IFSP_EXTREG_TP_INT)
			return paramComSize == 1 ? "i32"
									 : ("v" + std::to_string(paramComSize) + "i32");
		if (baseIdentifier == IFSP_EXTREG_TP_FLOAT)
			return paramComSize == 1 ? "f32"
									 : ("v" + std::to_string(paramComSize) + "f32");
		return "unknown";
	}
} // namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv

namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv::ExtInst::
	GlslStd450
{
#define DEF_INST(x)                                                              \
	std::string x(const std::vector<SpvVMExtRegistryTypeIdentifier>& paramTypes, \
		const std::vector<int>&										 paramComCnt)
#define GET_TYPE(x) getLlvmType(paramTypes[(x)], paramComCnt[(x)])
#define GET_TYPE_ANNO(x) getMangledType(paramTypes[(x)], paramComCnt[(x)])

	inline std::string glslstdTrigonometryFunc(
		const std::vector<SpvVMExtRegistryTypeIdentifier>& paramTypes,
		const std::vector<int>& paramComCnt, int funcId,
		const std::string& funcName)
	{
		auto specType = GET_TYPE(0), baseType = getLlvmType(paramTypes[0], 1);
		auto mangledSpecType = GET_TYPE_ANNO(0),
			 mangledBaseType = getMangledType(paramTypes[0], 1);
		std::string		  functionName = "GLSL.std.450_" + std::to_string(funcId);
		std::string		  arguments[1] = { "%x" };
		std::string		  mangledName = functionName + "_" + mangledSpecType;

		std::stringstream ps;
		ps << "define " << specType << " @" << mangledName << "(" << specType << " "
		   << arguments[0] << "){\n";
		if (paramComCnt[0] == 1)
		{
			ps << "%p = call " << specType << "@llvm." << funcName << "(" << specType
			   << " " << arguments[0] << ")\n";
			ps << "ret " << specType << " %p \n";
		}
		else
		{
			std::string lastname = "undef";
			for (int i = 0; i < paramComCnt[0]; i++)
			{
				ps << "%p_" << i << " = extractelement " << specType << " "
				   << arguments[0] << ", i32 " << i << "\n";
				ps << "%ps_" << i << " = call " << baseType << " @llvm." << funcName
				   << "." << mangledBaseType << "(" << baseType << " %p_" << i << ")\n";
				ps << "%ret_" << i << " = insertelement " << specType << " " << lastname
				   << ", " << baseType << " %ps_" << i << ", i32 " << i << "\n";
				lastname = "%ret_" + std::to_string(i);
			}
			ps << "ret " << specType << " %ret_" << paramComCnt[0] - 1 << "\n";
		}
		ps << "}\n";
		return ps.str();
	}

	inline std::string glslstdVectorNormalize(
		const std::vector<SpvVMExtRegistryTypeIdentifier>& paramTypes,
		const std::vector<int>& paramComCnt, int funcId,
		const std::string& funcName)
	{
		auto specType = GET_TYPE(0), baseType = getLlvmType(paramTypes[0], 1);
		auto mangledSpecType = GET_TYPE_ANNO(0),
			 mangledBaseType = getMangledType(paramTypes[0], 1);
		std::string		  functionName = "GLSL.std.450_" + std::to_string(funcId);
		std::string		  arguments[1] = { "%x" };
		std::string		  mangledName = functionName + "_" + mangledSpecType;

		std::stringstream ps;
		ps << "define " << specType << " @" << mangledName << "(" << specType << " "
		   << arguments[0] << "){\n";
		for (int i = 0; i < paramComCnt[0]; i++)
		{
			ps << "%p_" << i << " = extractelement " << specType << " " << arguments[0]
			   << ", i32 " << i << "\n";
			ps << "%psquared_" << i << " = fmul " << baseType << " %p_" << i << ", %p_"
			   << i << "\n";
			if (i == 1)
			{
				ps << "%sum_" << i << " = fadd " << baseType << " %psquared_" << i
				   << ", %psquared_" << i - 1 << "\n";
			}
			else if (i > 1)
			{
				ps << "%sum_" << i << " = fadd " << baseType << " %psquared_" << i
				   << ", %sum_" << i - 1 << "\n";
			}
		}
		// Sqrt
		ps << "%len = call " << baseType << " @llvm.sqrt." << mangledBaseType << "("
		   << baseType << " %sum_" << paramComCnt[0] - 1 << ")\n";
		std::string lastName = "undef";
		for (int i = 0; i < paramComCnt[0]; i++)
		{
			ps << "%ps_" << i << " = fdiv " << baseType << " %p_" << i << ", %len\n";
			ps << "%ret_" << i << " = insertelement " << specType << " " << lastName
			   << ", " << baseType << " %ps_" << i << ", i32 " << i << "\n";
			lastName = "%ret_" + std::to_string(i);
		}
		ps << "ret " << specType << " %ret_" << paramComCnt[0] - 1 << "\n";
		ps << "}\n";
		return ps.str();
	}
	inline std::string glslstdFusedMultiplyAdd(
		const std::vector<SpvVMExtRegistryTypeIdentifier>& paramTypes,
		const std::vector<int>& paramComCnt, int funcId,
		const std::string& funcName)
	{
		auto specType = GET_TYPE(0), baseType = getLlvmType(paramTypes[0], 1);
		auto mangledSpecType = GET_TYPE_ANNO(0),
			 mangledBaseType = getMangledType(paramTypes[0], 1);
		std::string		  functionName = "GLSL.std.450_" + std::to_string(funcId);
		std::string		  arguments[3] = { "%x", "%y", "%z" };
		std::string		  mangledName = functionName + "_" + mangledSpecType + "_" + mangledSpecType + "_" + mangledSpecType;

		std::stringstream ps;
		ps << "define " << specType << " @" << mangledName << "(" << specType << " "
		   << arguments[0] << ", " << specType << " " << arguments[1] << ", "
		   << specType << " " << arguments[2] << "){\n";
		if (paramComCnt[0] == 1)
		{
			ps << "%p = call " << specType << " @llvm.fma.f32(" << specType << " "
			   << arguments[0] << ", " << specType << " " << arguments[1] << ", "
			   << specType << " " << arguments[2] << ")\n";
			ps << "ret " << specType << " %p \n";
		}
		else
		{
			std::string lastname = "undef";
			for (int i = 0; i < paramComCnt[0]; i++)
			{
				ps << "%p_" << i << " = extractelement " << specType << " "
				   << arguments[0] << ", i32 " << i << "\n";
				ps << "%q_" << i << " = extractelement " << specType << " "
				   << arguments[1] << ", i32 " << i << "\n";
				ps << "%r_" << i << " = extractelement " << specType << " "
				   << arguments[2] << ", i32 " << i << "\n";
				ps << "%ps_" << i << " = call " << baseType << " @llvm.fma."
				   << mangledBaseType << "(" << baseType << " %p_" << i << ", "
				   << baseType << " %q_" << i << ", " << baseType << " %r_" << i << ")\n";
				ps << "%ret_" << i << " = insertelement " << specType << " " << lastname
				   << ", " << baseType << " %ps_" << i << ", i32 " << i << "\n";
				lastname = "%ret_" + std::to_string(i);
			}
			ps << "ret " << specType << " %ret_" << paramComCnt[0] - 1 << "\n";
		}
		ps << "}\n";
		return ps.str();
	}

	DEF_INST(Sin)
	{
		return glslstdTrigonometryFunc(paramTypes, paramComCnt, GLSLstd450Sin, "sin");
	}
	DEF_INST(Cos)
	{
		return glslstdTrigonometryFunc(paramTypes, paramComCnt, GLSLstd450Cos, "cos");
	}
	DEF_INST(Tan)
	{
		return glslstdTrigonometryFunc(paramTypes, paramComCnt, GLSLstd450Tan, "tan");
	}
	DEF_INST(Asin)
	{
		return glslstdTrigonometryFunc(paramTypes, paramComCnt, GLSLstd450Asin,
			"asin");
	}
	DEF_INST(Acos)
	{
		return glslstdTrigonometryFunc(paramTypes, paramComCnt, GLSLstd450Acos,
			"acos");
	}
	DEF_INST(Atan)
	{
		return glslstdTrigonometryFunc(paramTypes, paramComCnt, GLSLstd450Atan,
			"atan");
	}
	DEF_INST(Normalize)
	{
		return glslstdVectorNormalize(paramTypes, paramComCnt, GLSLstd450Normalize,
			"Normalize");
	}
	DEF_INST(Fma)
	{
		return glslstdFusedMultiplyAdd(paramTypes, paramComCnt, GLSLstd450Fma, "fma");
	}
#undef DEF_INST
} // namespace
  // Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv::ExtInst::GlslStd450

namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv
{
	SpvVMExtRegistry::SpvVMExtRegistry()
	{
#define REGISTER_GLSLSTD450(x, y) \
	generators["GLSL.std.450"][(x)] = (ExtInst::GlslStd450::y);
		REGISTER_GLSLSTD450(GLSLstd450::GLSLstd450Sin, Sin);
		REGISTER_GLSLSTD450(GLSLstd450::GLSLstd450Cos, Cos);
		REGISTER_GLSLSTD450(GLSLstd450::GLSLstd450Tan, Tan);
		REGISTER_GLSLSTD450(GLSLstd450::GLSLstd450Asin, Asin);
		REGISTER_GLSLSTD450(GLSLstd450::GLSLstd450Acos, Acos);
		REGISTER_GLSLSTD450(GLSLstd450::GLSLstd450Atan, Atan);
		REGISTER_GLSLSTD450(GLSLstd450::GLSLstd450Normalize, Normalize);
		REGISTER_GLSLSTD450(GLSLstd450::GLSLstd450Fma, Fma);

#undef REGISTER_GLSLSTD450
	}
	std::string SpvVMExtRegistry::queryExternalFunc(
		std::string extImportName, int functionName,
		const std::vector<SpvVMExtRegistryTypeIdentifier>& identifiers,
		const std::vector<int>&							   componentSize)
	{
		auto baseFuncName = extImportName + "_" + std::to_string(functionName);
		for (int i = 0; i < identifiers.size(); i++)
		{
			auto tpIdentifier = getMangledType(identifiers[i], componentSize[i]);
			baseFuncName += "_" + tpIdentifier;
		}
		// Lookup
		if (registeredFunc.count(baseFuncName) == 0)
		{
			registeredFunc.insert(baseFuncName);
			auto lpt1 = generators.count(extImportName);
			if (lpt1 == 0)
			{
				printf("OpExtInst: library %s not supported \n", extImportName.c_str());
				return "{error func}";
			}
			auto lpt2 = generators[extImportName].count(functionName);
			if (lpt2 == 0)
			{
				printf("OpExtInst: function %d not supported \n", functionName);
				return "{error func}";
			}
			auto genFunc = generators[extImportName][functionName];
			auto genIr = genFunc(identifiers, componentSize);
			irCode += genIr;
		}
		return baseFuncName;
	}
	std::string SpvVMExtRegistry::getRequiredFuncDefs()
	{
		return irCode;
	}
} // namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv