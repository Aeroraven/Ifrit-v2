#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include "iconfont/IconFontAwesome.h"

namespace Ifrit::Editor::Internal
{
    enum class EGameObjectType
    {
        Unspecified,
        Camera,
        Light,
        Mesh,
        SceneNode,
    };

    String GetComponentIcon(String typeName)
    {
        static HashMap<String, String> iconMap = {
            { "Transform", ICON_FA_COMPASS_DRAFTING },
            { "Camera", ICON_FA_CAMERA },
            { "MeshFilter", ICON_FA_CUBE },
            { "MeshRenderer", ICON_FA_CUBE },
            { "Light", ICON_FA_LIGHTBULB },
        };

        auto it = iconMap.find(typeName);
        if (it != iconMap.end())
        {
            return it->second;
        }
        return ICON_FA_GEARS;
    }

    String GetGameObjectIcon(EGameObjectType type)
    {
        switch (type)
        {
            case EGameObjectType::Camera:
                return ICON_FA_CAMERA;
            case EGameObjectType::Light:
                return ICON_FA_LIGHTBULB;
            case EGameObjectType::Mesh:
                return ICON_FA_CIRCLE_NODES;
            case EGameObjectType::SceneNode:
                return ICON_FA_LAYER_GROUP;
            case EGameObjectType::Unspecified:
            default:
                return ICON_FA_CUBE;
        }
    }

} // namespace Ifrit::Editor::Internal