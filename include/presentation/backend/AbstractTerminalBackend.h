#pragma once
#include "presentation/backend/BackendProvider.h"
#include "core/definition/CoreExports.h"
#include "core/data/Image.h"

namespace Ifrit::Presentation::Backend {
	class AbstractTerminalBackend :public BackendProvider{
	public:
		virtual void setCursor(int x, int y, std::string& str);
	};
}