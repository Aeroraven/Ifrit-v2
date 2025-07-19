#pragma once
#include "ifrit/core/base/IfritBase.h"

#ifndef IFRIT_EDITOR_SHARED_ASSET_PATH
    #error "IFRIT_EDITOR_SHARED_ASSET_PATH is not defined. Please define it in your build system."
    #define IFRIT_EDITOR_SHARED_ASSET_PATH ""
#endif

namespace Ifrit::Editor::Internal
{
    namespace AssetPath
    {
        inline constexpr const char* kDefaultFont = IFRIT_EDITOR_SHARED_ASSET_PATH "/Fonts/DejaVuSans.ttf";
    }

} // namespace Ifrit::Editor::Internal