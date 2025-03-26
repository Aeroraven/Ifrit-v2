
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

#include "ifrit/ircompile/llvm_ircompile.h"
#include "ifrit/softgraphics/engine/comllvmrt/WrappedLLVMRuntime.h"

namespace Ifrit::Graphics::SoftGraphics::ComLLVMRuntime
{
	struct WrappedLLVMRuntimeContext
	{
		IfritCompLLVMExecutionSession* session;
		std::string					   irCode;
		std::string					   irIdentifier;
	};

	WrappedLLVMRuntime::WrappedLLVMRuntime()
	{
		this->session = new WrappedLLVMRuntimeContext();
		session->session = nullptr;
	}

	WrappedLLVMRuntime::~WrappedLLVMRuntime()
	{
		if (session)
		{
			if (session->session)
				IfritCom_LlvmExec_Destroy(session->session);
			delete this->session;
		}
	}

	void WrappedLLVMRuntime::initLlvmBackend()
	{
		IfritCom_LlvmExec_Init();
	}

	void WrappedLLVMRuntime::loadIR(std::string irCode, std::string irIdentifier)
	{
		if (session->session)
		{
			IfritCom_LlvmExec_Destroy(session->session);
		}
		session->irCode = irCode;
		session->irIdentifier = irIdentifier;
		session->session =
			IfritCom_LlvmExec_Create(irCode.c_str(), irIdentifier.c_str());
	}

	void* WrappedLLVMRuntime::lookupSymbol(std::string symbol)
	{
		if (!session->session)
		{
			return nullptr;
		}
		return IfritCom_LlvmExec_Lookup(session->session, symbol.c_str());
	}

	std::unique_ptr<ShaderRuntime> WrappedLLVMRuntime::getThreadLocalCopy()
	{
		auto copy = std::make_unique<WrappedLLVMRuntime>();
		copy->loadIR(session->irCode, session->irIdentifier);
		return copy;
	}

	WrappedLLVMRuntimeBuilder::WrappedLLVMRuntimeBuilder()
	{
		WrappedLLVMRuntime::initLlvmBackend();
	}

	std::unique_ptr<ShaderRuntime> WrappedLLVMRuntimeBuilder::buildRuntime() const
	{
		return std::make_unique<WrappedLLVMRuntime>();
	}
} // namespace Ifrit::Graphics::SoftGraphics::ComLLVMRuntime