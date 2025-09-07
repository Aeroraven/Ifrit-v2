#include "ifrit.internal/editor/imgui/ImGuiUtilities.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/editor/widgets/FileDialog.h"
#include "ifrit/runtime/scene/SceneManager.h"
#include "ifrit/core/file/FileOps.h"
#include "imgui.h"
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

    // Add this function to handle the popup rendering (called every frame)
    IFRIT_APIDECL void MenuBar_RenderImportAssetPopup(MenuBar_Modals& config)
    {
        static char                            selectedFilePath[512] = "";
        static char                            selectedName[256]     = "NewAsset";
        static int                             selectedImporterIndex = 0;
        static Vec<Runtime::AssetImporterPair> importers;
        static bool                            importersLoaded = false;
        static bool                            alwaysOpen      = true;

        if (config.mAssetImporterPopupOpen)
        {
            ImGui::OpenPopup("Import Asset Popup");
            config.mAssetImporterPopupOpen = false;
        }

        // Load importers once
        if (!importersLoaded)
        {
            auto allImporters = Ifrit::Runtime::GetActiveApplication()->GetAssetRegistry()->GetAllImporters();
            importers         = allImporters;
            importersLoaded   = true;
        }

        if (ImGui::BeginPopupModal("Import Asset Popup", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            // File selection area
            ImGui::Text("File to Import:");
            ImGui::SameLine();

            ImGui::PushItemWidth(300.0f);
            ImGui::InputText("##filepath", selectedFilePath, sizeof(selectedFilePath), ImGuiInputTextFlags_ReadOnly);
            ImGui::PopItemWidth();

            ImGui::SameLine();
            if (ImGui::Button("Browse..."))
            {
                Widget::FFileDialogSetupArgs args;
                args.mDialogType = Widget::EFileDialogType::OpenFile;
                args.mFileTypes.push_back({ "All Files", "*" });
                auto ret = config.mFileDialog->OpenFileDialog(args);
                if (ret.mSuccess)
                {
                    strncpy_s(selectedFilePath, ret.mFilePath.c_str(), sizeof(selectedFilePath) - 1);
                    selectedFilePath[sizeof(selectedFilePath) - 1] = '\0';
                }
            }

            ImGui::Spacing();

            // Name input area
            ImGui::Text("Asset Name:");
            ImGui::SameLine();
            ImGui::PushItemWidth(300.0f);
            ImGui::InputText("##assetname", selectedName, sizeof(selectedName));
            ImGui::PopItemWidth();

            if (strlen(selectedName) == 0)
            {
                strncpy_s(selectedName, "NewAsset", sizeof(selectedName) - 1);
                selectedName[sizeof(selectedName) - 1] = '\0';
            }

            ImGui::Spacing();

            // Importer selection dropdown
            ImGui::Text("Importer:");
            ImGui::SameLine();

            ImGui::PushItemWidth(300.0f);
            if (ImGui::BeginCombo("##importer",
                    importers.empty() ? "No importers available"
                                      : importers[selectedImporterIndex].mImporterId.c_str()))
            {
                for (int i = 0; i < importers.size(); ++i)
                {
                    bool isSelected = (selectedImporterIndex == i);
                    if (ImGui::Selectable(importers[i].mImporterId.c_str(), isSelected))
                    {
                        selectedImporterIndex = i;
                    }
                    if (isSelected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::PopItemWidth();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // OK and Cancel buttons
            bool canImport =
                strlen(selectedFilePath) > 0 && !importers.empty() && selectedImporterIndex < importers.size();

            if (!canImport)
            {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button("OK", ImVec2(80, 0)))
            {
                if (canImport)
                {
                    auto selectedImporter  = importers[selectedImporterIndex].mImporter;
                    auto selectedFile      = selectedFilePath;
                    auto selectedAssetName = selectedName;
                    auto assetManager      = Ifrit::Runtime::GetActiveApplication()->GetAssetRegistry();

                    auto result = assetManager->ImportAssetFromAbsPath(
                        importers[selectedImporterIndex].mImporterId, selectedFile, selectedAssetName);

                    if (result.mCode == Ifrit::Runtime::EAssetRegistrationResultCode::Success)
                    {
                        IF_LOG_INFO("ImGuiUtilities", "Asset imported successfully: {}", selectedAssetName);
                    }
                    else
                    {
                        IF_LOG_ERROR("ImGuiUtilities", "Failed to import asset: {}. Error code: {}", selectedAssetName,
                            static_cast<int>(result.mCode));
                    }

                    // Reset state and close popup
                    selectedFilePath[0]   = '\0';
                    selectedImporterIndex = 0;
                    ImGui::CloseCurrentPopup();
                }
            }

            if (!canImport)
            {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(80, 0)))
            {
                // Reset state and close popup
                selectedFilePath[0]   = '\0';
                selectedImporterIndex = 0;
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
        alwaysOpen = true; // Keep the popup open by default
    }

    // This function is called when menu item is clicked (opens the popup)
    IFRIT_APIDECL void MenuBar_ImportAsset(MenuBar_Modals& config)
    {
        config.mAssetImporterPopupOpen = true;
        IF_LOG_INFO("ImGuiUtilities", "Opening Import Asset popup");
    }
} // namespace Ifrit::Editor::ImGuiInternal