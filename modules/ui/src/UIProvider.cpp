#include "ifrit/ui/UIProvider.h"

namespace Ifrit::UI
{
    IFRIT_APIDECL void UIProvider::OnInitialize(Runtime::IApplication* app) { m_Application = app; }
    IFRIT_APIDECL void UIProvider::OnShutdown() { m_Application = nullptr; }
    IFRIT_APIDECL void UIProvider::OnFrameBegin() {}
    IFRIT_APIDECL void UIProvider::OnFrameEnd() {}

} // namespace Ifrit::UI