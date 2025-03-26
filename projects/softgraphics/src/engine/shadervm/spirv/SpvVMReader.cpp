
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

#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMReader.h"
#include <spirv_headers/include/spirv/unified1/spirv.hpp>

namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv
{
	std::vector<char> SpvVMReader::readFile(const char* path)
	{
		std::vector<char> data;
		FILE*			  file = fopen(path, "rb");
		if (!file)
		{
			ifritError("Failed to open file:", std::string(path));
		}
		fseek(file, 0, SEEK_END);
		size_t length = ftell(file);
		fseek(file, 0, SEEK_SET);
		data.resize(length);
		fread(data.data(), 1, length, file);
		fclose(file);
		return data;
	}
	void SpvVMReader::initializeContext(SpvVMContext* outContext)
	{
		int v = 1;
		outContext->endianParserNative =
			((*(char*)(&v)) == 1) ? IFSP_LITTLE_ENDIAN : IFSP_BIG_ENDIAN;
	}
	void SpvVMReader::parseByteCode(const char* byteCode, size_t length,
		SpvVMContext* outContext)
	{
		// https://github.com/KhronosGroup/SPIRV-Guide/blob/main/chapters/parsing_instructions.md
		if (length < 5)
		{
			ifritError("Invalid SPIR-V data: too short");
		}
		bool doBswap = false;
		auto readWord = [&](const uint32_t*& x) {
			auto val = *x;
			x++;
			if (doBswap)
				val =
					Ifrit::Graphics::SoftGraphics::Core::Utility::byteSwapU32(val);
			return val;
		};
		const uint32_t*		  pCur = reinterpret_cast<const uint32_t*>(byteCode);
		const uint32_t* const pStart = pCur;
		outContext->headerMagic = *pCur;
		pCur++;
		if (outContext->headerMagic == spv::MagicNumber)
			outContext->endianBytecode = outContext->endianParserNative;
		else
			(doBswap = true), outContext->endianBytecode = (outContext->endianParserNative == IFSP_LITTLE_ENDIAN) ? IFSP_BIG_ENDIAN : IFSP_LITTLE_ENDIAN;

		outContext->headerVersion = readWord(pCur);
		outContext->headerGenerator = readWord(pCur);
		outContext->headerBound = readWord(pCur);
		outContext->headerSchema = readWord(pCur);
		while (pCur - pStart < static_cast<uint32_t>(length))
		{
			uint32_t			opWord = readWord(pCur);
			SpvVMCtxInstruction opIns;
			opIns.opCode = opWord & spv::OpCodeMask;
			opIns.opWordCounts = opWord >> spv::WordCountShift;
			if (pCur - 1 + opIns.opWordCounts > pStart + length || opIns.opWordCounts == 0)
			{
				ifritError("Invalid SPIR-V data: corrupted instruction, at position",
					pCur - pStart);
			}
			opIns.opParams.resize(opIns.opWordCounts - 1);
			for (auto i = 0u; i < opIns.opWordCounts - 1; i++)
			{
				opIns.opParams[i] = readWord(pCur);
			}
			outContext->instructions.emplace_back(std::move(opIns));
		}
	}
	void SpvVMReader::printParsedInstructions(SpvVMContext* outContext)
	{
		for (int i = 0; const auto& ins : outContext->instructions)
		{
			std::cout << "Instruction:" << i++ << ": " << ins.opCode << " | ";
			for (const auto& j : ins.opParams)
			{
				printf("[%d]", j);
			}
			printf("\n");
		}
	}
} // namespace Ifrit::Graphics::SoftGraphics::ShaderVM::Spirv