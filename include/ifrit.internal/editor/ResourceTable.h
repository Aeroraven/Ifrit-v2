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
        inline IF_CONSTEXPR const char* kDefaultFont   = IFRIT_EDITOR_SHARED_ASSET_PATH "/Fonts/DejaVuSans.ttf";
        inline IF_CONSTEXPR const char* kDefaultFAFont = IFRIT_EDITOR_SHARED_ASSET_PATH "/Fonts/FaSolid.ttf";
    } // namespace AssetPath

} // namespace Ifrit::Editor::Internal