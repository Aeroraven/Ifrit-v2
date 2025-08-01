#pragma once
#include "ifrit/editor/EditorBase.h"
#include "ifrit/core/algo/Guid.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/core/reflection/Reflection.h"
namespace Ifrit::Editor::ImGuiInternal
{
    struct Inspector_NewSceneNodeModal
    {
        char                mNewNodeName[114];
        bool                mPopupOpen  = false;
        Runtime::SceneNode* mParentNode = nullptr;
    };

    struct Inspector_AddGameObjectModal
    {
        char                mNewGameObjectName[114];
        bool                mPopupOpen  = false;
        Runtime::SceneNode* mParentNode = nullptr;
    };

    struct Inspector_ComponentCreationPopupConfig
    {
        char                                 NewComponentName[128] = "";
        String                               SelectedComponent;
        const Reflection::FReflTypeMetaInfo* SelectedComponentMeta = nullptr;
        Runtime::GameObject*                 mTargetGameObject     = nullptr;
        bool                                 PopupOpen             = false;
    };

    struct Inspector_Modals
    {
        Inspector_NewSceneNodeModal            mAddSceneNodeModal;
        Inspector_AddGameObjectModal           mAddGameObjectModal;
        Inspector_ComponentCreationPopupConfig mComponentCreationPopup;
        float                                  mDpiScaler = 1.0f;
    };

    IFRIT_EDITOR_API void Inspector_ShowGameObjectProperties(
        Runtime::GameObject* gameObject, GUID activeGuid, Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_ShowSceneNodeProperties(
        Runtime::SceneNode* node, GUID activeGuid, Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_ShowAddSceneNodeModal(Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_ShowAddGameObjectModal(Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_ShowComponentCreationPopup(Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_RegisterEdit1ingHandles();
} // namespace Ifrit::Editor::ImGuiInternal
