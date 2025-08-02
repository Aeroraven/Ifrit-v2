#include "ifrit.internal/editor/imgui/ImGuiUtilities.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/editor/widgets/FileDialog.h"
#include "ifrit/runtime/scene/SceneManager.h"
#include "ifrit/core/file/FileOps.h"
namespace Ifrit::Editor::ImGuiInternal
{
    IFRIT_APIDECL void MenuBar_ExportCurrentScene(MenuBar_Modals& config)
    {
        Widget::FFileDialogSetupArgs args;
        args.mFileTypes.push_back({ "Scene Files", "ifritscene" });
        auto ret = config.mFileDialog->OpenFileDialog(args);
        if (ret.mSuccess)
        {
            auto activeScene = Ifrit::Runtime::GetActiveApplication()->GetSceneManager()->GetActiveScene();
            if (activeScene)
            {
                auto serialized = activeScene->Serialize();
                WriteTextFile(ret.mFilePath, serialized);
                IF_LOG_INFO("ImGuiUtilities", "Scene exported to {}", ret.mFilePath);
            }
        }
    }
    IFRIT_APIDECL void MenuBar_LoadAndOverrideCurrentScene(MenuBar_Modals& config)
    {
        Widget::FFileDialogSetupArgs args;
        args.mDialogType = Widget::EFileDialogType::OpenFile;
        args.mFileTypes.push_back({ "Scene Files", "ifritscene" });
        auto ret = config.mFileDialog->OpenFileDialog(args);
        if (ret.mSuccess)
        {
            auto activeScene = Ifrit::Runtime::GetActiveApplication()->GetSceneManager()->GetActiveScene();
            if (activeScene)
            {
                activeScene->Deserialize(ReadTextFile(ret.mFilePath));
                IF_LOG_INFO("ImGuiUtilities", "Scene loaded and overridden from {}", ret.mFilePath);
            }
        }
    }
} // namespace Ifrit::Editor::ImGuiInternal