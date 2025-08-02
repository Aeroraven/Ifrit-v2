#pragma once
#include "ifrit/editor/EditorBase.h"
#include "ifrit/core/algo/Guid.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/core/reflection/Reflection.h"
#include "ifrit/editor/widgets/FileDialog.h"

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
        bool                mPopupOpen    = false;
        bool                mGPUTransform = false;
        Runtime::SceneNode* mParentNode   = nullptr;
    };

    struct Inspector_ComponentCreationPopupConfig
    {
        char                                 NewComponentName[128] = "";
        String                               SelectedComponent;
        const Reflection::FReflTypeMetaInfo* SelectedComponentMeta = nullptr;
        Runtime::GameObject*                 mTargetGameObject     = nullptr;
        bool                                 PopupOpen             = false;
        bool                                 NewComponentEnabled   = true;
    };

    struct Inspector_Modals
    {
        Inspector_NewSceneNodeModal            mAddSceneNodeModal;
        Inspector_AddGameObjectModal           mAddGameObjectModal;
        Inspector_ComponentCreationPopupConfig mComponentCreationPopup;
        float                                  mDpiScaler = 1.0f;
        Widget::FileDialog*                    mFileDialog;
    };

    struct MenuBar_Modals
    {
        Widget::FileDialog* mFileDialog;
    };

    IFRIT_EDITOR_API void Inspector_RegisterEditingHandles();
    IFRIT_EDITOR_API void Inspector_ShowGameObjectProperties(
        Runtime::GameObject* gameObject, GUID activeGuid, Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_ShowSceneNodeProperties(
        Runtime::SceneNode* node, GUID activeGuid, Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_ShowAddSceneNodeModal(Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_ShowAddGameObjectModal(Inspector_Modals& config);
    IFRIT_EDITOR_API void Inspector_ShowComponentCreationPopup(Inspector_Modals& config);

    IFRIT_EDITOR_API void MenuBar_ExportCurrentScene(MenuBar_Modals& config);
    IFRIT_EDITOR_API void MenuBar_LoadAndOverrideCurrentScene(MenuBar_Modals& config);
} // namespace Ifrit::Editor::ImGuiInternal
