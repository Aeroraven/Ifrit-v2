#include "ifrit/editor/imgui/ImGuiProvider.h"
#include "ifrit.internal/editor/imgui/ImGuiStyling.h"
#include "ifrit.internal/editor/IconMapping.h"
#include "ifrit.internal/editor/ResourceTable.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/runtime/base/Property.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/base/ActorBehavior.h"
#include "ifrit/runtime/base/Camera.h"
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/runtime/base/MeshComponent.h"
#include "ifrit/core/typing/Rtti.h"

#include "glfw/glfw3.h"
#include "ifrit/core/hal/HalDisplay.h"
#include "iconfont/IconFontAwesome.h"
#include "ifrit/runtime/renderer/SharedRenderResource.h"
#include "ifrit/runtime/base/ActorBehavior.h"
#include "ifrit/core/hal/HalWindow.h"
#include "ifrit/core/reflection/Reflection.h"
#include "ifrit/runtime/application/ApplicationState.h"

#include "ifrit.internal/editor/imgui/ImGuiUtilities.h"

#include "ifrit/editor/widgets/FileDialog.h"

namespace Ifrit::Editor
{
    struct ImGuiProviderData
    {
        Vec<String>                     m_RegisteredGameObjects;
        Vec<GUID>                       m_RegisteredGameObjectsUUID;
        i32                             m_SelectedGameObjectIndex = -1;
        GUID                            m_ActiveGameObjectUUID;

        VkDescriptorSet                 m_EditorSceneView = VK_NULL_HANDLE;
        ImGuiID                         m_DockspaceID     = 0;
        bool                            m_IsDockingSetup  = false;

        f32                             m_DpiScaler       = 1.0f;
        ImGuiInternal::Inspector_Modals m_InspectorModals = {};
        ImGuiInternal::MenuBar_Modals   m_MenuBarModals   = {};
        ImFont*                         m_IconFontLarge   = nullptr;

        Widget::FileDialog              mFileDialog;
    };

    static VkFormat imguiColorAttachmentFormats[] = { VK_FORMAT_B8G8R8A8_SRGB };

    static void     imguiCheckResultFn(VkResult err)
    {
        if (err == VK_SUCCESS)
            return;
        IF_LOG_ASSERTION("Editor.ImGui", false, "ImGui Vulkan error: {}", (int)err);
    }

    void ShowAssetTreeView(ImGuiProviderData* data)
    {
        auto assetManager = Runtime::GetActiveApplication()->GetAssetRegistry();
        auto assets       = assetManager->GetAllAssetMetadata();

        // Categorize assets by type
        HashMap<Runtime::EAssetType, Vec<Runtime::AssetMetadata>> categorizedAssets;

        for (const auto& asset : assets)
        {
            categorizedAssets[asset.mAssetType].push_back(asset);
        }

        // Define asset type names and icons
        auto getAssetTypeName = [](Runtime::EAssetType type) -> String {
            switch (type)
            {
                case Runtime::EAssetType::General:
                    return "General";
                case Runtime::EAssetType::Texture:
                    return "Textures";
                case Runtime::EAssetType::Material:
                    return "Materials";
                case Runtime::EAssetType::Mesh:
                    return "Meshes";
                case Runtime::EAssetType::Prefab:
                    return "Prefabs";
                case Runtime::EAssetType::Shader:
                    return "Shaders";
                case Runtime::EAssetType::VolumetricData:
                    return "Volumes";
                default:
                    return "Unknown";
            }
        };

        auto getAssetTypeIcon = [](Runtime::EAssetType type) -> const char* {
            switch (type)
            {
                case Runtime::EAssetType::General:
                    return ICON_FA_FILE;
                case Runtime::EAssetType::Texture:
                    return ICON_FA_IMAGE;
                case Runtime::EAssetType::Material:
                    return ICON_FA_PALETTE;
                case Runtime::EAssetType::Mesh:
                    return ICON_FA_CUBE;
                case Runtime::EAssetType::Prefab:
                    return ICON_FA_OBJECT_GROUP;
                case Runtime::EAssetType::Shader:
                    return ICON_FA_CODE;
                case Runtime::EAssetType::VolumetricData:
                    return ICON_FA_DATABASE;
                default:
                    return ICON_FA_QUESTION;
            }
        };

        auto getAssetIcon = [](Runtime::EAssetType type) -> const char* {
            switch (type)
            {
                case Runtime::EAssetType::Texture:
                    return ICON_FA_FILE_IMAGE;
                case Runtime::EAssetType::Material:
                    return ICON_FA_FILL_DRIP;
                case Runtime::EAssetType::Mesh:
                    return ICON_FA_VECTOR_SQUARE;
                case Runtime::EAssetType::Prefab:
                    return ICON_FA_SHAPES;
                case Runtime::EAssetType::Shader:
                    return ICON_FA_FILE_CODE;
                case Runtime::EAssetType::VolumetricData:
                    return ICON_FA_DATABASE;
                default:
                    return ICON_FA_FILE;
            }
        };

        // Render tree view for each category
        for (const auto& [assetType, assetList] : categorizedAssets)
        {
            if (assetList.empty())
                continue;

            String             categoryName = String(getAssetTypeIcon(assetType)) + " " + getAssetTypeName(assetType);
            String             categoryId   = "Category_" + std::to_string(static_cast<int>(assetType));

            ImGuiTreeNodeFlags categoryFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

            // Check if this category should be expanded by default
            if (assetType == Runtime::EAssetType::Texture || assetType == Runtime::EAssetType::Mesh)
            {
                categoryFlags |= ImGuiTreeNodeFlags_DefaultOpen;
            }

            bool categoryOpen = ImGui::TreeNodeEx(
                categoryId.c_str(), categoryFlags, "%s (%zu)", categoryName.c_str(), assetList.size());

            if (categoryOpen)
            {
                // Render assets in this category
                for (const auto& asset : assetList)
                {
                    ImGui::PushID(asset.mGuid.ToString().c_str());

                    String             assetDisplayName = String(getAssetIcon(assetType)) + " " + asset.mName;

                    ImGuiTreeNodeFlags assetFlags =
                        ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Bullet;

                    // if (data->m_SelectedAssetGUID == asset.mGuid)
                    // {
                    //     assetFlags |= ImGuiTreeNodeFlags_Selected;
                    // }

                    ImGui::TreeNodeEx(assetDisplayName.c_str(), assetFlags);

                    // Handle asset selection
                    if (ImGui::IsItemClicked())
                    {
                        // data->m_SelectedAssetGUID = asset.mGuid;
                        // data->m_SelectedAssetType = asset.mAssetType;

                        // Log or handle asset selection
                        IF_LOG_INFO(
                            "Editor.Assets", "Selected asset: {} (Type: {})", asset.mName, getAssetTypeName(assetType));
                    }

                    // Right-click context menu
                    if (ImGui::BeginPopupContextItem())
                    {
                        if (ImGui::MenuItem("Open"))
                        {
                            // Handle asset opening
                            IF_LOG_INFO("Editor.Assets", "Opening asset: {}", asset.mName);
                        }
                        if (ImGui::MenuItem("Delete"))
                        {
                            // Handle asset deletion
                            IF_LOG_INFO("Editor.Assets", "Deleting asset: {}", asset.mName);
                        }
                        if (ImGui::MenuItem("Rename"))
                        {
                            // Handle asset renaming
                            IF_LOG_INFO("Editor.Assets", "Renaming asset: {}", asset.mName);
                        }
                        ImGui::Separator();
                        if (ImGui::MenuItem("Show in Explorer"))
                        {
                            // Handle showing asset in file explorer
                            IF_LOG_INFO("Editor.Assets", "Showing asset in explorer: {}", asset.mName);
                        }
                        ImGui::EndPopup();
                    }

                    ImGui::PopID();
                }

                ImGui::TreePop();
            }
        }
    }

    void ShowAssetGridWithIcons(ImGuiProviderData* data, float iconFontSize, float padding)
    {
        auto  assetManager = Runtime::GetActiveApplication()->GetAssetRegistry();
        auto  assets       = assetManager->GetAllAssetMetadata();

        // Adjust column width to be larger
        float columnWidth = iconFontSize + padding * 2.0f;

        // Calculate the number of columns based on the available width
        int   columns = static_cast<int>(ImGui::GetContentRegionAvail().x / columnWidth);
        if (columns < 1)
            columns = 1;

        ImGui::Columns(columns, nullptr, false); // Create columns for the grid

        for (const auto& asset : assets)
        {
            ImGui::PushID(asset.mGuid.ToString().c_str()); // Unique ID for each asset

            // Center-align the column content
            float columnStartX  = ImGui::GetCursorPosX();
            float columnCenterX = columnStartX + (columnWidth / 2.0f);

            // Render the icon (use a large font for the icon)
            ImGui::PushFont(data->m_IconFontLarge);
            float iconWidth = ImGui::CalcTextSize(ICON_FA_FILE).x;
            ImGui::SetCursorPosX(columnCenterX - (iconWidth / 2.0f)); // Center the icon
            ImGui::TextUnformatted(ICON_FA_FILE);                     // Replace with your icon character
            ImGui::PopFont();

            // Render the text (truncate if necessary)
            String displayName = asset.mName;
            float  textWidth   = ImGui::CalcTextSize(displayName.c_str()).x;
            if (textWidth > columnWidth - padding * 2.0f) // Truncate if text is too wide
            {
                size_t maxChars = static_cast<size_t>(
                    (columnWidth - padding * 2.0f - ImGui::CalcTextSize("...").x) / ImGui::CalcTextSize("A").x);
                if (maxChars > 0 && maxChars < asset.mName.size())
                {
                    displayName = asset.mName.substr(0, maxChars) + "...";
                    textWidth   = ImGui::CalcTextSize(displayName.c_str()).x;
                }
            }

            ImGui::SetCursorPosX(columnCenterX - (textWidth / 2.0f)); // Center the text
            ImGui::TextUnformatted(displayName.c_str());

            ImGui::NextColumn(); // Move to the next column
            ImGui::PopID();
        }

        ImGui::Columns(1); // Reset columns
    }

    void PrintingLogs()
    {
        VecView<Logging::InternalLogEntries> entries = Logging::GetLogEntries();
        ImVec4                               color   = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

        // Check if we're at or very close to the bottom (with small tolerance)
        bool                                 wasAtBottom = false;
        float                                scrollY     = ImGui::GetScrollY();
        float                                maxScrollY  = ImGui::GetScrollMaxY();
        if (scrollY >= maxScrollY - 5.0f) // 5 pixel tolerance
        {
            wasAtBottom = true;
        }

        for (auto& p : entries)
        {
            String tx = p.m_Message;
            switch (p.m_Level)
            {
                case Logging::ELoggingLevel::Critical:
                    tx    = "[CRITICAL] " + tx;
                    color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
                    break;
                case Logging::ELoggingLevel::Error:
                    tx    = "[ERROR   ] " + tx;
                    color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
                    break;
                case Logging::ELoggingLevel::Warning:
                    tx    = "[WARNING ] " + tx;
                    color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                    break;
                case Logging::ELoggingLevel::Info:
                    tx    = "[INFO    ] " + tx;
                    color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
                    break;
                case Logging::ELoggingLevel::Debug:
                    tx    = "[DEBUG   ] " + tx;
                    color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                    break;
                case Logging::ELoggingLevel::Trace:
                    tx    = "[TRACE   ] " + tx;
                    color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                    break;
                default:
                    tx    = "[UNKNOWN ] " + tx;
                    color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
                    break;
            }
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::Text(tx.c_str());
            ImGui::PopStyleColor();
        }

        // Only auto-scroll if we were at the bottom before rendering
        if (wasAtBottom)
        {
            ImGui::SetScrollHereY(1.0f);
        }
    }

    void RenderMenuBar(ImGuiProviderData* data)
    {
        // Remove ImGui::BeginMainMenuBar() and ImGui::EndMainMenuBar()
        // Use ImGui::BeginMenuBar() instead since we're inside a window with MenuBar flag
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Import Asset"))
                {
                    ImGuiInternal::MenuBar_ImportAsset(data->m_MenuBarModals);
                }
                if (ImGui::MenuItem("Export Current Scene"))
                {
                    ImGuiInternal::MenuBar_ExportCurrentScene(data->m_MenuBarModals);
                }

                if (ImGui::MenuItem("Load And Override Scene"))
                {
                    ImGuiInternal::MenuBar_LoadAndOverrideCurrentScene(data->m_MenuBarModals);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Profiler"))
            {
                if (ImGui::MenuItem("Frame Capture"))
                {
                    auto app                                                 = Runtime::GetActiveApplication();
                    app->GetApplicationState()->mProfilerRequestFrameCapture = true;
                    IF_LOG_INFO("Editor.ImGui", "Renderdoc capture requested for next frame.");
                }

                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
    }

    static VkPipelineRenderingCreateInfoKHR GetImGuiRenderingInfo()
    {
        VkPipelineRenderingCreateInfoKHR info = {};
        info.sType                            = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
        info.pNext                            = nullptr;
        info.colorAttachmentCount             = 1;
        info.pColorAttachmentFormats          = imguiColorAttachmentFormats;
        info.depthAttachmentFormat            = VK_FORMAT_UNDEFINED;
        info.stencilAttachmentFormat          = VK_FORMAT_UNDEFINED;
        info.viewMask                         = 0;
        return info;
    }

    void RenderGameObjectTreeView(ImGuiProviderData* data, Runtime::Scene* scene)
    {
        // Clear the registered objects as we'll rebuild them during traversal
        data->m_RegisteredGameObjects.clear();
        data->m_RegisteredGameObjectsUUID.clear();

        i32      currentIndex = 0;
        Vec<i32> nodeStack; // Track which nodes need to be popped

        scene->DepthFirstTraverse(
            [&](Runtime::SceneNode* node) {
                if (node)
                {
                    String nodeName = node->GetName();
                    String displayName =
                        Internal::GetGameObjectIcon(Internal::EGameObjectType::SceneNode) + " " + nodeName;
                    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

                    // Check if node has children or game objects
                    bool hasChildren = node->GetChildren().size() > 0 || node->GetGameObjects().size() > 0;
                    if (!hasChildren)
                    {
                        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                    }

                    bool nodeOpen = ImGui::TreeNodeEx(displayName.c_str(), flags);
                    if (ImGui::IsItemClicked())
                    {
                        data->m_SelectedGameObjectIndex = currentIndex;
                        data->m_ActiveGameObjectUUID    = node->GetGUID();
                    }
                    bool needsPop = nodeOpen && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen);
                    nodeStack.push_back(needsPop);
                    return needsPop;
                }
                else
                {
                    nodeStack.push_back(false); // No node created, no pop needed
                }
                return false;
            },

            [&](Runtime::GameObject* obj) {
                if (obj)
                {
                    String                    objectName = obj->GetName();
                    GUID                      objectUUID = obj->GetUUID();

                    Internal::EGameObjectType objectIconType = Internal::EGameObjectType::Unspecified;

                    if (obj->GetComponent<Runtime::Camera>())
                    {
                        objectIconType = Internal::EGameObjectType::Camera;
                    }
                    else if (obj->GetComponent<Runtime::MeshFilter>())
                    {
                        objectIconType = Internal::EGameObjectType::Mesh;
                    }

                    String objectIcon  = Internal::GetGameObjectIcon(objectIconType);
                    String displayText = objectIcon + " " + objectName;

                    // Add to our tracking lists
                    data->m_RegisteredGameObjects.push_back(objectName);
                    data->m_RegisteredGameObjectsUUID.push_back(objectUUID);

                    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                    if (data->m_ActiveGameObjectUUID == objectUUID)
                    {
                        flags |= ImGuiTreeNodeFlags_Selected;
                    }

                    ImGui::TreeNodeEx(displayText.c_str(), flags);
                    if (ImGui::IsItemClicked())
                    {
                        data->m_SelectedGameObjectIndex = currentIndex;
                        data->m_ActiveGameObjectUUID    = objectUUID;
                    }

                    currentIndex++;
                }
            },

            []() {
                // fnOnPush - do nothing
            },

            [&]() {
                // fnOnPop - only pop if we actually need to
                if (!nodeStack.empty())
                {
                    bool needsPop = nodeStack.back();
                    nodeStack.pop_back();
                    if (needsPop)
                    {
                        ImGui::TreePop();
                    }
                }
            });
    }

    IFRIT_APIDECL ImGuiProvider::ImGuiProvider() : Super(), m_Data(new ImGuiProviderData()) {}
    IFRIT_APIDECL ImGuiProvider::~ImGuiProvider()
    {
        delete m_Data;
        m_Data = nullptr;
    }

    IFRIT_APIDECL void ImGuiProvider::OnInitialize(Runtime::IApplication* app)
    {

        Super::OnInitialize(app);
        auto displayProvider   = app->GetDisplay();
        auto windowObject      = displayProvider->GetGLFWWindow();
        auto appState          = app->GetApplicationState();
        appState->m_EditorMode = true;

        auto projectProperty = app->GetProjectProperty();
        auto rhi             = app->GetRhi();
        auto dq              = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Graphics);
        IF_LOG_ASSERTION("Editor.ImGui", projectProperty.m_rhiType == Runtime::AppRhiType::Vulkan,
            "ImGuiProvider only supports Vulkan RHI type");
        IF_LOG_ASSERTION("Editor.ImGui", projectProperty.m_displayProvider == Runtime::AppDisplayProvider::GLFW,
            "ImGuiProvider only supports GLFW display provider");

        if (projectProperty.m_EnableDPIScaling)
        {
            m_Data->m_DpiScaler                  = HAL::GetDisplayScale();
            m_Data->m_InspectorModals.mDpiScaler = m_Data->m_DpiScaler;
        }

        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

        io.DisplaySize.x = static_cast<f32>(projectProperty.m_width * m_Data->m_DpiScaler);
        io.DisplaySize.y = static_cast<f32>(projectProperty.m_height * m_Data->m_DpiScaler);

        // C:/Windows/Fonts/NotoSans-Regular.ttf
        io.Fonts->AddFontFromFileTTF(Internal::AssetPath::kDefaultFont, 12.0f * m_Data->m_DpiScaler, nullptr);

        ImFontConfig config;
        config.MergeMode        = true;
        config.GlyphMinAdvanceX = 13.0f;
        io.Fonts->AddFontFromFileTTF(Internal::AssetPath::kDefaultFAFont, 12.0f * m_Data->m_DpiScaler, &config);
        // large font
        config.MergeMode = false;
        m_Data->m_IconFontLarge =
            io.Fonts->AddFontFromFileTTF(Internal::AssetPath::kDefaultFAFont, 48.0f * m_Data->m_DpiScaler, &config);

        // ImGui::StyleColorsDark();
        Internal::ApplyImGuiSytle();

        ImGuiStyle& style = ImGui::GetStyle();
        style.ScaleAllSizes(m_Data->m_DpiScaler);

        ImGui_ImplGlfw_InitForVulkan(reinterpret_cast<GLFWwindow*>(windowObject), true);

        ImGui_ImplVulkan_InitInfo init_info   = {};
        init_info.ApiVersion                  = VK_API_VERSION_1_3;
        init_info.Instance                    = reinterpret_cast<VkInstance>(rhi->GetRawHandle_Instance());
        init_info.PhysicalDevice              = reinterpret_cast<VkPhysicalDevice>(rhi->GetRawHandle_ActiveAdapter());
        init_info.Device                      = reinterpret_cast<VkDevice>(rhi->GetRawHandle_Device());
        init_info.QueueFamily                 = dq->GetRawHandle_Family();
        init_info.Queue                       = reinterpret_cast<VkQueue>(dq->GetRawHandle());
        init_info.MinImageCount               = projectProperty.m_rhiNumBackBuffers;
        init_info.ImageCount                  = projectProperty.m_rhiNumBackBuffers;
        init_info.MSAASamples                 = VK_SAMPLE_COUNT_1_BIT;
        init_info.UseDynamicRendering         = true;
        init_info.PipelineRenderingCreateInfo = GetImGuiRenderingInfo();
        init_info.DescriptorPoolSize          = 114514;
        init_info.CheckVkResultFn             = imguiCheckResultFn;

        ImGui_ImplVulkan_Init(&init_info);

        ImGuiInternal::Inspector_RegisterEditingHandles();

        m_Data->mFileDialog.Initialize();
        m_Data->m_InspectorModals.mFileDialog = &m_Data->mFileDialog;
        m_Data->m_MenuBarModals.mFileDialog   = &m_Data->mFileDialog;
    }

    IFRIT_APIDECL void ImGuiProvider::OnShutdown()
    {
        Super::OnShutdown();
        m_Data->mFileDialog.Finalize();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    IFRIT_APIDECL void ImGuiProvider::OnFrameBegin()
    {
        ImGuiIO& io      = ImGui::GetIO();
        io.DisplaySize.x = static_cast<f32>(m_Application->GetProjectProperty().m_width * m_Data->m_DpiScaler);
        io.DisplaySize.y = static_cast<f32>(m_Application->GetProjectProperty().m_height * m_Data->m_DpiScaler);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    IFRIT_APIDECL void ImGuiProvider::OnFrameEnd() {}

    IFRIT_APIDECL void ImGuiProvider::OnUpdate(Runtime::Scene* scene)
    {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        // Main Menu Bar
        RenderMenuBar(m_Data);

        m_Data->m_DockspaceID = ImGui::GetID("MainDockSpace");
        ImGui::DockSpace(m_Data->m_DockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

        if (!m_Data->m_IsDockingSetup)
        {
            ImGui::DockBuilderRemoveNode(m_Data->m_DockspaceID);
            ImGui::DockBuilderAddNode(m_Data->m_DockspaceID, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(m_Data->m_DockspaceID, ImGui::GetMainViewport()->Size);

            ImGuiID dock_left, dock_right, dock_left_bottom, dock_bottom;
            ImGuiID dock_main = m_Data->m_DockspaceID;

            dock_left        = ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Left, 0.15f, nullptr, &dock_main);
            dock_right       = ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Right, 0.25f, nullptr, &dock_main);
            dock_left_bottom = ImGui::DockBuilderSplitNode(dock_left, ImGuiDir_Down, 0.5f, nullptr, &dock_left);
            dock_bottom      = ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Down, 0.27f, nullptr, &dock_main);

            // Dock windows to specific areas
            ImGui::DockBuilderDockWindow("Asset", dock_left);                  // Asset explorer - left top
            ImGui::DockBuilderDockWindow("Scene Hierarchy", dock_left_bottom); // Scene hierarchy - left bottom
            ImGui::DockBuilderDockWindow("Inspector", dock_right);             // Inspector - right side
            ImGui::DockBuilderDockWindow("Viewport", dock_main);               // Viewport - center
            ImGui::DockBuilderDockWindow("Console", dock_bottom);              // Console - bottom
            ImGui::DockBuilderDockWindow("Profiling", dock_bottom);

            // Finish setup
            ImGui::DockBuilderFinish(m_Data->m_DockspaceID);
            m_Data->m_IsDockingSetup = true;
        }
        ImGui::End();

        ImGui::Begin("Viewport");
        if (m_Data->m_EditorSceneView == VK_NULL_HANDLE)
        {
            auto texture = m_Application->GetDefaultColorImage();
            auto sampler = reinterpret_cast<VkSampler>(
                m_Application->GetSharedRenderResource()->GetLinearClampSampler()->GetRawHandle());
            auto texView              = reinterpret_cast<VkImageView>(texture->GetRawHandle_DefaultView());
            auto texLayout            = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            m_Data->m_EditorSceneView = ImGui_ImplVulkan_AddTexture(sampler, texView, texLayout);
        }
        ImVec2 windowSize               = ImGui::GetWindowSize();
        ImVec2 contentRegionAvail       = ImGui::GetContentRegionAvail();
        ImVec2 windowPos                = ImGui::GetCursorScreenPos();
        auto   appState                 = m_Application->GetApplicationState();
        auto   displayProvider          = m_Application->GetDisplay();
        appState->mEditorViewportX      = windowPos.x - displayProvider->GetWindowLeft();
        appState->mEditorViewportY      = windowPos.y - displayProvider->GetWindowTop();
        appState->mEditorViewportWidth  = contentRegionAvail.x;
        appState->mEditorViewportHeight = contentRegionAvail.y;

        ImGui::Image(m_Data->m_EditorSceneView, contentRegionAvail, ImVec2(0, 0), ImVec2(1, 1));
        ImGui::End();

        ImGui::Begin("Inspector");
        auto objects    = scene->FilterObjects([](auto) { return true; });
        auto sceneNodes = scene->FilterNodes([](auto) { return true; });
        m_Data->m_RegisteredGameObjects.clear();
        m_Data->m_RegisteredGameObjectsUUID.clear();
        for (auto& node : sceneNodes)
        {
            auto UUID = node->GetGUID();
            m_Data->m_RegisteredGameObjects.push_back(node->GetName());
            m_Data->m_RegisteredGameObjectsUUID.push_back(UUID);
            ImGuiInternal::Inspector_ShowSceneNodeProperties(
                node, m_Data->m_ActiveGameObjectUUID, m_Data->m_InspectorModals);
        }
        for (auto& obj : objects)
        {
            auto UUID = obj->GetUUID();
            m_Data->m_RegisteredGameObjects.push_back(obj->GetName());
            m_Data->m_RegisteredGameObjectsUUID.push_back(UUID);
            ImGuiInternal::Inspector_ShowGameObjectProperties(
                obj, m_Data->m_ActiveGameObjectUUID, m_Data->m_InspectorModals);
        }
        ImGui::End();

        ImGui::Begin("Scene Hierarchy");
        RenderGameObjectTreeView(m_Data, scene);
        ImGui::End();

        ImGui::Begin("Asset");
        // ShowAssetGridWithIcons(m_Data, 96.0f * m_Data->m_DpiScaler, 8.0f * m_Data->m_DpiScaler);
        ShowAssetTreeView(m_Data);
        ImGui::End();

        ImGui::Begin("Console");
        PrintingLogs();
        ImGui::End();

        ImGui::Begin("Profiling");
        ImGuiInternal::Profiler_ShowGPUScopeStats();
        ImGui::End();

        // Modals
        ImGuiInternal::Inspector_ShowAddSceneNodeModal(m_Data->m_InspectorModals);
        ImGuiInternal::Inspector_ShowAddGameObjectModal(m_Data->m_InspectorModals);
        ImGuiInternal::Inspector_ShowComponentCreationPopup(m_Data->m_InspectorModals);
        ImGuiInternal::MenuBar_RenderImportAssetPopup(m_Data->m_MenuBarModals);
    }

    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> ImGuiProvider::OnPostRendering(RHI::RhiTaskSubmission* prevSubmission)
    {

        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();

        auto        rhi            = m_Application->GetRhi();
        auto        dq             = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Graphics);
        auto        swapchainImage = rhi->GetSwapchainImage();
        auto        swapchainView  = reinterpret_cast<VkImageView>(swapchainImage->GetRawHandle_DefaultView());

        // http://zhuanlan.zhihu.com/p/293067607
        ImGuiIO&    io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        return dq->RunAsyncCommand(
            [draw_data, swapchainView, this](const RHI::RhiCommandList* cmd) {
                cmd->BeginScope("Ifrit.UI: ImGui Rendering");
                auto     vkcmd = reinterpret_cast<VkCommandBuffer>(cmd->GetRawHandle());

                VkRect2D renderArea;
                renderArea.extent = { (u32)draw_data->DisplaySize.x, (u32)draw_data->DisplaySize.y };
                renderArea.offset = { 0, 0 };

                VkRenderingAttachmentInfo colorAttachmentInfo{};
                colorAttachmentInfo.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                colorAttachmentInfo.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                colorAttachmentInfo.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentInfo.imageView   = swapchainView;
                colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;

                VkRenderingInfo renderingInfo{};
                renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
                renderingInfo.renderArea           = renderArea;
                renderingInfo.layerCount           = 1;
                renderingInfo.colorAttachmentCount = 1;
                renderingInfo.pColorAttachments    = &colorAttachmentInfo;
                renderingInfo.pDepthAttachment     = nullptr;

                RHI::RhiResourceBarrier texBarrier;
                texBarrier.m_type                     = RHI::RhiBarrierType::Transition;
                texBarrier.m_transition.m_type        = RHI::RhiResourceType::Texture;
                texBarrier.m_transition.m_texture     = m_Application->GetDefaultColorImage();
                texBarrier.m_transition.m_srcState    = RHI::RhiResourceState::AutoTraced;
                texBarrier.m_transition.m_dstState    = RHI::RhiResourceState::ShaderRead;
                texBarrier.m_transition.m_subResource = { 0, 0, 1, 1 };

                RHI::RhiResourceBarrier swapchainBarrier;
                swapchainBarrier.m_type                     = RHI::RhiBarrierType::Transition;
                swapchainBarrier.m_transition.m_type        = RHI::RhiResourceType::Texture;
                swapchainBarrier.m_transition.m_texture     = m_Application->GetRhi()->GetSwapchainImage();
                swapchainBarrier.m_transition.m_srcState    = RHI::RhiResourceState::AutoTraced;
                swapchainBarrier.m_transition.m_dstState    = RHI::RhiResourceState::ColorRT;
                swapchainBarrier.m_transition.m_subResource = { 0, 0, 1, 1 };
                cmd->AddResourceBarrier({ texBarrier, swapchainBarrier });

                vkCmdBeginRendering(vkcmd, &renderingInfo);
                if (draw_data->Valid)
                {
                    ImGui_ImplVulkan_RenderDrawData(draw_data, vkcmd);
                }
                else
                {
                    IF_LOG_WARNING("Editor.ImGui", "ImGui draw data is not valid, skipping rendering.");
                }
                vkCmdEndRendering(vkcmd);
                texBarrier.m_transition.m_srcState = RHI::RhiResourceState::AutoTraced;
                texBarrier.m_transition.m_dstState = RHI::RhiResourceState::ColorRT;
                cmd->AddResourceBarrier({ texBarrier });

                cmd->EndScope();
            },
            { prevSubmission }, {});
    }
} // namespace Ifrit::Editor
