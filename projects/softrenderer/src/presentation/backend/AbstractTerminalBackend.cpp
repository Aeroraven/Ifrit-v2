#include "presentation/backend/AbstractTerminalBackend.h"

namespace Ifrit::Presentation::Backend {
	void AbstractTerminalBackend::setCursor(int x, int y, std::string& str) {
		str = "\033[" + std::to_string(y) + ";" + std::to_string(x) + "H" + str;
	}
}