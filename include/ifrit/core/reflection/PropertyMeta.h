#pragma once
#include "ifrit/core/base/IfritBase.h"
namespace Ifrit::Reflection
{
    enum class EPropertyEditable
    {
        None,
        Editable,
        ReadOnly,
    };

    enum class EPropertyUIControl
    {
        None,
        UISlider,
        UISelect,
        UIText,
        UIColor,
    };

    enum class EPropertyAssetCategory : u8
    {
        Generic,
        Texture,
        Material,
        Mesh,
        Shader,
        Scene,
        Audio,
        Script,
        Font,
        Animation,
        Prefab
    };
    struct PropertyMetadata
    {
        String                 Name;
        EPropertyEditable      Editable      = EPropertyEditable::None;
        EPropertyUIControl     UIControl     = EPropertyUIControl::None;
        EPropertyAssetCategory AssetCategory = EPropertyAssetCategory::Generic;
        f64                    UIClampMin    = 0.0;
        f64                    UIClampMax    = 1.0;
    };

} // namespace Ifrit::Reflection
