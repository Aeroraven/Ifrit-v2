#include "ifrit/runtime/base/ApplicationInterface.h"

namespace Ifrit::Runtime
{

    static IApplication*            sActiveApplication = nullptr;

    IFRIT_RUNTIME_API IApplication* GetActiveApplication() { return sActiveApplication; }
    void                            SetActiveApplication(IApplication* app) { sActiveApplication = app; }

} // namespace Ifrit::Runtime