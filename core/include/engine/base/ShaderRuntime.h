#include "core/definition/CoreExports.h"
namespace Ifrit::Engine {
	class ShaderRuntime {
	public:
		virtual void loadIR(std::string shaderCode, std::string shaderIdentifier) = 0;
		virtual void* lookupSymbol(std::string symbol) = 0;
		virtual std::unique_ptr<ShaderRuntime> getThreadLocalCopy() = 0;
	};
}