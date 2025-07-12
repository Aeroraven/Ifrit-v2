#pragma once
#include "ifrit/core/altina/AlBase.h"
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::Altina
{

    IFRIT_APIDECL void AlErrorImpl(const char* message) { iError("Alina: {}", message); }

} // namespace Ifrit::Altina