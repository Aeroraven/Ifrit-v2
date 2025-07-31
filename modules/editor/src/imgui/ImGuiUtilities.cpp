#include "ifrit.internal/editor/imgui/ImGuiUtilities.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit.internal/editor/IconMapping.h"
#include "ifrit/core/typing/Rtti.h"
#include "ifrit/core/reflection/Reflection.h"

using namespace Ifrit::Runtime;

namespace Ifrit::Editor::ImGuiInternal
{
    template <typename T>
    static void GeneralSelectableHandle(const char* label, T& value, const Vec<Pair<T, String>>& options)
    {
        String previewValue = "(Invalid)";
        for (const auto& option : options)
        {
            if (option.first == value)
            {
                previewValue = option.second;
                break;
            }
        }
        if (ImGui::BeginCombo(label, previewValue.c_str()))
        {
            for (const auto& option : options)
            {
                bool isSelected = (value == option.first);
                if (ImGui::Selectable(option.second.c_str(), isSelected))
                {
                    value = option.first;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    static void RegisterSingleEditorHandle(const char* name, Fn<bool()> predicate, Fn<void()> handle)
    {
        ImGui::Text("%s", name);
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-FLT_MIN);
        bool isActive = predicate();
        if (!isActive)
        {
            ImGui::BeginDisabled();
        }
        handle();
        if (!isActive)
        {
            ImGui::EndDisabled();
        }
    }

    IFRIT_EDITOR_API void Inspector_RegisterEditingHandles()
    {
        // Aux
        auto& handles           = Reflection::GetPropertyUIAuxHandles();
        handles.m_OnPreRegister = []() {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
        };
        handles.m_OnPostRegister = []() {};

        // Func
        auto& funcHandles             = Reflection::GetFunctionUIHandle();
        funcHandles.mFunctionCallback = [](const char* name, Fn<void()> func) {
            ImGui::PushItemWidth(-1);
            float buttonWidth = ImGui::GetContentRegionAvail().x; // Get available width
            ImGui::PushStyleVar(
                ImGuiStyleVar_FramePadding, ImVec2(0, 5)); // Optional: Adjust padding for better appearance
            if (ImGui::Button(name, ImVec2(buttonWidth, 0)))
            {
                func();
            }
            ImGui::PopStyleVar();
            ImGui::PopItemWidth();
        };

        // Float32
        auto& f32Handles           = Reflection::GetPropertyUIHandle<f32>();
        f32Handles.mSliderCallback = [](const char* name, f32& value, f32 min, f32 max, Fn<bool()> predicate) {
            RegisterSingleEditorHandle(name, predicate, [&]() {
                ImGui::SliderFloat(
                    (String("##") + name).c_str(), &value, min, max, "%.3f", ImGuiSliderFlags_AlwaysClamp);
            });
        };
        f32Handles.mTextCallback = [](const char* name, f32& value, Fn<bool()> predicate) {
            RegisterSingleEditorHandle(name, predicate,
                [&]() { ImGui::InputFloat((String("##") + name).c_str(), &value, 0.0f, 0.0f, "%.3f"); });
        };

        // Int32
        auto& i32Handles           = Reflection::GetPropertyUIHandle<i32>();
        i32Handles.mSliderCallback = [](const char* name, i32& value, i32 min, i32 max, Fn<bool()> predicate) {
            RegisterSingleEditorHandle(
                name, predicate, [&]() { ImGui::SliderInt((String("##") + name).c_str(), &value, min, max); });
        };

        i32Handles.mTextCallback = [](const char* name, i32& value, Fn<bool()> predicate) {
            RegisterSingleEditorHandle(
                name, predicate, [&]() { ImGui::InputInt((String("##") + name).c_str(), &value); });
        };

        // Vector3f
        auto& v3fHandles         = Reflection::GetPropertyUIHandle<Vector3f>();
        v3fHandles.mTextCallback = [](const char* name, Vector3f& value, Fn<bool()> predicate) {
            RegisterSingleEditorHandle(
                name, predicate, [&]() { ImGui::InputFloat3((String("##") + name).c_str(), &value.x, "%.4f"); });
        };

        // Vector4f
        auto& v4fHandles         = Reflection::GetPropertyUIHandle<Vector4f>();
        v4fHandles.mTextCallback = [](const char* name, Vector4f& value, Fn<bool()> predicate) {
            RegisterSingleEditorHandle(
                name, predicate, [&]() { ImGui::InputFloat4((String("##") + name).c_str(), &value.x, "%.4f"); });
        };

        v4fHandles.mColorCallback = [](const char* name, Vector4f& value, Fn<bool()> predicate) {
            RegisterSingleEditorHandle(name, predicate,
                [&]() { ImGui::ColorEdit4((String("##") + name).c_str(), &value.x, ImGuiColorEditFlags_NoInputs); });
        };

        // Int8
        auto& i8Handles           = Reflection::GetPropertyUIHandle<i8>();
        i8Handles.mSelectCallback = [](const char* name, i8& value, Vec<Pair<i8, String>> options,
                                        Fn<bool()> predicate) {
            RegisterSingleEditorHandle(
                name, predicate, [&]() { GeneralSelectableHandle<i8>((String("##") + name).c_str(), value, options); });
        };

        // UInt8
        auto& u8Handles           = Reflection::GetPropertyUIHandle<u8>();
        u8Handles.mSelectCallback = [](const char* name, u8& value, Vec<Pair<u8, String>> options,
                                        Fn<bool()> predicate) {
            RegisterSingleEditorHandle(
                name, predicate, [&]() { GeneralSelectableHandle<u8>((String("##") + name).c_str(), value, options); });
        };

        // Bool
        auto& boolHandles           = Reflection::GetPropertyUIHandle<bool>();
        boolHandles.mSelectCallback = [](const char* name, bool& value, Vec<Pair<bool, String>> options,
                                          Fn<bool()> predicate) {
            RegisterSingleEditorHandle(name, predicate, [&]() {
                auto Id = (String("##") + name).c_str();
                ImGui::PushID(Id);
                ImGui::PushID(&value);
                ImGui::Checkbox(Id, &value);
                ImGui::PopID();
                ImGui::PopID();
            });
        };
    }

    IFRIT_EDITOR_API void Inspector_ShowSceneNodeProperties(
        Runtime::SceneNode* node, GUID activeGuid, Inspector_Modals& config)
    {
        auto obj  = node;
        auto UUID = obj->GetGUID();
        if (UUID == activeGuid)
        {
            ImGui::Text("%s (SceneNode)", obj->GetName().c_str());
            ImGui::TextDisabled("UUID: %s", obj->GetGUID().ToString().c_str());
            ImGui::Separator();

            // Calculate available width
            float buttonWidth = ImGui::GetContentRegionAvail().x;

            // Make the button use full width
            if (ImGui::Button("Add Child Node", ImVec2(buttonWidth, 0)))
            {
                config.mAddSceneNodeModal.mParentNode = node;
                config.mAddSceneNodeModal.mPopupOpen  = true;
            }
            if (ImGui::Button("Add Game Object", ImVec2(buttonWidth, 0)))
            {
                config.mAddGameObjectModal.mParentNode = node;
                config.mAddGameObjectModal.mPopupOpen  = true;
            }
        }
    }

    IFRIT_EDITOR_API void Inspector_ShowAddSceneNodeModal(Inspector_Modals& config)
    {
        if (config.mAddSceneNodeModal.mPopupOpen)
        {
            ImGui::OpenPopup("Add Scene Node");
            config.mAddSceneNodeModal.mPopupOpen = false;
        }

        if (ImGui::BeginPopupModal("Add Scene Node", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            // Apply DPI scaling
            ImGui::SetNextWindowSize(ImVec2(300 * config.mDpiScaler, 200 * config.mDpiScaler), ImGuiCond_Always);

            ImGui::Text("Parent Node: %s",
                config.mAddSceneNodeModal.mParentNode ? config.mAddSceneNodeModal.mParentNode->GetName().c_str()
                                                      : "None");
            ImGui::Separator();

            ImGui::InputText("Node Name", config.mAddSceneNodeModal.mNewNodeName,
                sizeof(config.mAddSceneNodeModal.mNewNodeName),
                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);

            if (ImGui::Button("Create"))
            {
                if (config.mAddSceneNodeModal.mParentNode)
                {
                    config.mAddSceneNodeModal.mParentNode->AddChildNode(String(config.mAddSceneNodeModal.mNewNodeName));
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel"))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    IFRIT_EDITOR_API void Inspector_ShowAddGameObjectModal(Inspector_Modals& config)
    {
        if (config.mAddGameObjectModal.mPopupOpen)
        {
            ImGui::OpenPopup("Add GameObject");
            config.mAddGameObjectModal.mPopupOpen = false;
        }
        if (ImGui::BeginPopupModal("Add GameObject", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            // Apply DPI scaling
            ImGui::SetNextWindowSize(ImVec2(300 * config.mDpiScaler, 200 * config.mDpiScaler), ImGuiCond_Always);

            ImGui::Text("Parent Node: %s",
                config.mAddGameObjectModal.mParentNode ? config.mAddGameObjectModal.mParentNode->GetName().c_str()
                                                       : "None");
            ImGui::Separator();

            ImGui::InputText("GameObject Name", config.mAddGameObjectModal.mNewGameObjectName,
                sizeof(config.mAddGameObjectModal.mNewGameObjectName),
                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);

            if (ImGui::Button("Create"))
            {
                if (config.mAddGameObjectModal.mParentNode)
                {
                    config.mAddGameObjectModal.mParentNode->AddGameObject(
                        String(config.mAddGameObjectModal.mNewGameObjectName));
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel"))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    IFRIT_EDITOR_API void Inspector_ShowComponentCreationPopup(Inspector_Modals& configTop)
    {
        auto& config = configTop.mComponentCreationPopup;
        if (config.PopupOpen)
        {
            // Apply DPI scaling to the popup size
            ImGui::SetNextWindowSize(ImVec2(500 * configTop.mDpiScaler, 400 * configTop.mDpiScaler), ImGuiCond_Always);

            ImGui::OpenPopup("Add Component");
            config.PopupOpen = false;
        }

        if (ImGui::BeginPopupModal("Add Component", nullptr, ImGuiWindowFlags_NoResize))
        {
            // Fetch all derived component types
            auto derivedTypes = Ifrit::Reflection::GetAllDerivedTypes<Runtime::Component>(false);

            // Sort the derived types by their fully qualified name
            std::sort(derivedTypes.begin(), derivedTypes.end(),
                [](const auto& a, const auto& b) { return String{ a.get().Name } < String{ b.get().Name }; });

            // Build a hierarchical namespace tree
            struct NamespaceNode
            {
                std::unordered_map<std::string, NamespaceNode>    Children;
                std::vector<std::string>                          Components;
                std::vector<const Reflection::FReflTypeMetaInfo*> TypeMetas;
            };

            NamespaceNode root;

            for (const auto& typeMeta : derivedTypes)
            {
                std::string fullName = String{ typeMeta.get().Name }; // Fully qualified name (namespace::classname)

                // Split the full name into namespace hierarchy and class name
                std::vector<std::string> parts;
                size_t                   start = 0;
                size_t                   separatorPos;
                while ((separatorPos = fullName.find("::", start)) != std::string::npos)
                {
                    parts.push_back(fullName.substr(start, separatorPos - start));
                    start = separatorPos + 2; // Skip "::"
                }
                parts.push_back(fullName.substr(start)); // Add the class name

                // Insert into the namespace tree
                NamespaceNode* currentNode = &root;
                for (size_t i = 0; i < parts.size() - 1; ++i)
                {
                    currentNode = &currentNode->Children[parts[i]];
                }
                currentNode->Components.push_back(parts.back());
                currentNode->TypeMetas.push_back(&typeMeta.get());
            }

            // Recursive function to display the namespace tree with icons
            std::function<void(const NamespaceNode&, const std::string&)> displayTree =
                [&](const NamespaceNode& node, const std::string& namespaceName) {
                    for (const auto& [childName, childNode] : node.Children)
                    {
                        std::string fullNamespace =
                            namespaceName.empty() ? childName : namespaceName + "::" + childName;

                        // Add an icon to the namespace node
                        std::string label = std::string(ICON_FA_FOLDER) + " " + childName;

                        if (ImGui::TreeNode(label.c_str()))
                        {
                            displayTree(childNode, fullNamespace);
                            ImGui::TreePop();
                        }
                    }
                    for (auto i = 0; i < node.Components.size(); ++i)
                    {

                        const auto& componentName = node.Components[i];
                        const auto& componentMeta = node.TypeMetas[i];

                        std::string fullName =
                            namespaceName.empty() ? componentName : namespaceName + "::" + componentName;

                        // Add an icon to the component node
                        std::string        label = std::string(ICON_FA_CUBE) + " " + componentName;

                        // Determine if this component is selected
                        bool               isSelected = (config.SelectedComponent == fullName);

                        // Use TreeNodeEx to render the component node
                        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                        if (isSelected)
                        {
                            flags |= ImGuiTreeNodeFlags_Selected;
                            IF_LOG_INFO("Editor", "Selected component: {}", fullName);
                        }

                        if (ImGui::TreeNodeEx(label.c_str(), flags))
                        {
                            if (ImGui::IsItemClicked())
                            {
                                config.SelectedComponent     = fullName;
                                config.SelectedComponentMeta = componentMeta;
                                IF_LOG_INFO("Editor", "Selected component: {}", fullName);
                            }
                        }
                    }
                };

            // Display the tree view for component selection
            ImGui::Text("Select a Component:");
            ImGui::Separator();

            // Calculate available height dynamically and scale it
            float availableHeight =
                ImGui::GetContentRegionAvail().y - (50 * configTop.mDpiScaler); // Reserve space for buttons
            ImGui::BeginChild("TreeView", ImVec2(0, availableHeight), true, ImGuiWindowFlags_HorizontalScrollbar);
            displayTree(root, "");
            ImGui::EndChild();

            ImGui::Separator();

            // Input field for the component name
            ImGui::InputText("Component Name", config.NewComponentName, sizeof(config.NewComponentName));

            // Buttons for confirmation or cancellation
            if (ImGui::Button("Create"))
            {
                if (!config.SelectedComponent.empty() && strlen(config.NewComponentName) > 0)
                {
                    auto go = config.mTargetGameObject;
                    if (go)
                    {
                        go->AddComponentFromeMeta(config.SelectedComponentMeta->MetaInfo.GetMetaInfo());
                    }
                    ImGui::CloseCurrentPopup();
                }
                else
                {
                    ImGui::TextColored(ImVec4(1, 0, 0, 1), "Please select a component and enter a name.");
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel"))
            {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
    }

    IFRIT_EDITOR_API void Inspector_ShowGameObjectProperties(
        GameObject* gameObject, GUID activeGuid, Inspector_Modals& config)
    {
        auto obj  = gameObject;
        auto UUID = obj->GetUUID();
        if (UUID == activeGuid)
        {
            ImGui::Text("%s (GameObject)", obj->GetName().c_str());
            ImGui::TextDisabled("UUID: %s", obj->GetUUID().ToString().c_str());
            ImGui::Separator();

            float buttonWidth = ImGui::GetContentRegionAvail().x;

            // Make the button use full width
            if (ImGui::Button("Add Component", ImVec2(buttonWidth, 0)))
            {
                config.mComponentCreationPopup.mTargetGameObject = gameObject;
                config.mComponentCreationPopup.PopupOpen         = true;
                config.mComponentCreationPopup.SelectedComponent = "";
            }

            auto components = obj->GetAllComponents();
            for (auto& component : components)
            {
                auto   typeName      = GetDynamicTypeNameWithoutNamespace(component);
                auto   typeNamespace = GetDynamicTypeNamespace(component);
                auto   typeIcon      = Internal::GetComponentIcon(typeName);

                String displayHeader = typeIcon + " " + typeName;

                auto   isComponentMenuOpen = ImGui::CollapsingHeader(displayHeader.c_str());
                ImGui::SetItemTooltip("Component Namespace: %s", typeNamespace.c_str());
                if (isComponentMenuOpen)
                {
                    float availableWidth = ImGui::GetContentRegionAvail().x;
                    ImGui::PushID(component->GetGUID().ToString().c_str());
                    auto maxRows   = component->GetNumVisibleProperties() - 3;
                    f32  rowHeight = ImGui::GetTextLineHeightWithSpacing();
                    f32  maxHeight = rowHeight * maxRows;
                    if (ImGui::BeginTable("##properties", 2,
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp,
                            ImVec2(availableWidth, maxHeight)))
                    {
                        ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch, 0.4f);
                        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 0.6f);
                        ImGui::PushItemWidth(-1);
                        component->CallPropertyEditorHandle();
                        ImGui::PopItemWidth();
                        ImGui::EndTable();
                    }
                    component->CallFunctionEditorHandle();

                    ImGui::PopID();
                }
            }
        }
    }
} // namespace Ifrit::Editor::ImGuiInternal
