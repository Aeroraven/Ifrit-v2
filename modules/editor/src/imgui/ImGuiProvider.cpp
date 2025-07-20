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

#include "ifrit/runtime/base/ActorBehavior.h"
#include "ifrit/core/hal/HalWindow.h"

#define IMGUI_API IFRIT_APIDECL_IMPORT

namespace Ifrit::Editor
{
    struct ImGuiProviderData
    {
        Vec<String>     m_RegisteredGameObjects;
        Vec<String>     m_RegisteredGameObjectsUUID;
        i32             m_SelectedGameObjectIndex = -1;
        String          m_ActiveGameObjectUUID;

        VkDescriptorSet m_EditorSceneView = VK_NULL_HANDLE;
        ImGuiID         m_DockspaceID     = 0;
        bool            m_IsDockingSetup  = false;

        f32             m_DpiScaler = 1.0f;
    };

    static VkFormat imguiColorAttachmentFormats[] = { VK_FORMAT_B8G8R8A8_SRGB };

    static void     imguiCheckResultFn(VkResult err)
    {
        if (err == VK_SUCCESS)
            return;
        IF_LOG_ASSERTION("Editor.ImGui", false, "ImGui Vulkan error: {}", (int)err);
    }

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

    void PrintingLogs()
    {
        VecView<Logging::InternalLogEntries> entries = Logging::GetLogEntries();
        ImVec4                               color   = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
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
        ImGui::SetScrollHereY(1.0f);
    }

    static void RegisterEditorHandles(ImGuiProvider* provider)
    {
        // Aux
        auto& handles           = Runtime::GetPropertyEditorAxuHandles();
        handles.m_OnPreRegister = []() {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
        };
        handles.m_OnPostRegister = []() {};

        // Float32
        auto& f32Handles            = Runtime::GetPropertyEditorHandle<f32>();
        f32Handles.m_SliderCallback = [](const char* name, f32& value, f32 min, f32 max, f32 step) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::SliderFloat((String("##") + name).c_str(), &value, min, max, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        };
        f32Handles.m_TextCallback = [](const char* name, f32& value) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputFloat((String("##") + name).c_str(), &value, 0.0f, 0.0f, "%.3f");
        };

        // Int32
        auto& i32Handles            = Runtime::GetPropertyEditorHandle<i32>();
        i32Handles.m_SliderCallback = [](const char* name, i32& value, i32 min, i32 max, i32 step) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::SliderInt((String("##") + name).c_str(), &value, min, max);
        };

        i32Handles.m_TextCallback = [](const char* name, i32& value) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::InputInt((String("##") + name).c_str(), &value);
        };

        // Vector3f
        auto& v3fHandles          = Runtime::GetPropertyEditorHandle<Vector3f>();
        v3fHandles.m_TextCallback = [](const char* name, Vector3f& value) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputFloat3((String("##") + name).c_str(), &value.x, "%.4f");
        };

        // Vector4f
        auto& v4fHandles          = Runtime::GetPropertyEditorHandle<Vector4f>();
        v4fHandles.m_TextCallback = [](const char* name, Vector4f& value) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputFloat4((String("##") + name).c_str(), &value.x, "%.4f");
        };

        v4fHandles.m_ColorCallback = [](const char* name, Vector4f& value) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::ColorEdit4(
                (String("##") + name).c_str(), &value.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
        };

        // Int8
        auto& i8Handles            = Runtime::GetPropertyEditorHandle<i8>();
        i8Handles.m_SelectCallback = [](const char* name, i8& value, Vec<Pair<i8, String>> options) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            GeneralSelectableHandle<i8>(name, value, options);
        };

        // UInt8
        auto& u8Handles            = Runtime::GetPropertyEditorHandle<u8>();
        u8Handles.m_SelectCallback = [](const char* name, u8& value, Vec<Pair<u8, String>> options) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            GeneralSelectableHandle<u8>((String("##") + name).c_str(), value, options);
        };

        // Bool
        auto& boolHandles            = Runtime::GetPropertyEditorHandle<bool>();
        boolHandles.m_SelectCallback = [](const char* name, bool& value, Vec<Pair<bool, String>> options) {
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            auto Id = (String("##") + name).c_str();
            ImGui::PushID(Id);
            ImGui::PushID(&value);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::Checkbox(Id, &value);
            ImGui::PopID();
            ImGui::PopID();
        };
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

    void RenderGameObjectListBox(ImGuiProviderData* data)
    {
        if (ImGui::BeginListBox("##GameObjectList"))
        {
            for (size_t i = 0; i < data->m_RegisteredGameObjects.size(); ++i)
            {
                const bool isSelected = (data->m_SelectedGameObjectIndex == (i32)i);
                if (ImGui::Selectable(data->m_RegisteredGameObjects[i].c_str(), isSelected))
                {
                    data->m_SelectedGameObjectIndex = (i32)i;
                    data->m_ActiveGameObjectUUID    = data->m_RegisteredGameObjectsUUID[i];
                }
            }
            ImGui::EndListBox();
        }
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
                    String nodeName = "SceneNode";
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
                    String                    objectUUID = obj->GetUUID();

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
            m_Data->m_DpiScaler = HAL::GetDisplayScale();
        }

        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        io.DisplaySize.x           = projectProperty.m_width * m_Data->m_DpiScaler;
        io.DisplaySize.y           = projectProperty.m_height * m_Data->m_DpiScaler;

        // C:/Windows/Fonts/NotoSans-Regular.ttf
        io.Fonts->AddFontFromFileTTF(Internal::AssetPath::kDefaultFont, 12.0f * m_Data->m_DpiScaler, nullptr);

        ImFontConfig config;
        config.MergeMode        = true;
        config.GlyphMinAdvanceX = 13.0f;
        io.Fonts->AddFontFromFileTTF(Internal::AssetPath::kDefaultFAFont, 12.0f * m_Data->m_DpiScaler, &config);

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

        RegisterEditorHandles(this);
    }

    IFRIT_APIDECL void ImGuiProvider::OnShutdown()
    {
        Super::OnShutdown();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    IFRIT_APIDECL void ImGuiProvider::OnFrameBegin()
    {
        ImGuiIO& io      = ImGui::GetIO();
        io.DisplaySize.x = m_Application->GetProjectProperty().m_width * m_Data->m_DpiScaler;
        io.DisplaySize.y = m_Application->GetProjectProperty().m_height * m_Data->m_DpiScaler;

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
        m_Data->m_DockspaceID = ImGui::GetID("MainDockSpace");
        ImGui::DockSpace(m_Data->m_DockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

        if (!m_Data->m_IsDockingSetup)
        {
            ImGui::DockBuilderRemoveNode(m_Data->m_DockspaceID);
            ImGui::DockBuilderAddNode(m_Data->m_DockspaceID, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(m_Data->m_DockspaceID, ImGui::GetMainViewport()->Size);

            ImGuiID dock_left, dock_down, dock_log;
            ImGuiID dock_main = m_Data->m_DockspaceID;
            dock_left         = ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Left, 0.3f, nullptr, &dock_main);
            dock_down         = ImGui::DockBuilderSplitNode(dock_left, ImGuiDir_Down, 0.5f, nullptr, &dock_left);
            dock_log          = ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Down, 0.25f, nullptr, &dock_main);

            // Dock windows to specific areas
            ImGui::DockBuilderDockWindow("Scene Hierarchy", dock_down);
            ImGui::DockBuilderDockWindow("Inspector", dock_left);
            ImGui::DockBuilderDockWindow("Viewport", dock_main);
            ImGui::DockBuilderDockWindow("Console", dock_log);

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
        ImVec2 windowSize = ImGui::GetContentRegionAvail();
        ImGui::Image(m_Data->m_EditorSceneView, ImVec2(windowSize.x, windowSize.y), ImVec2(0, 0), ImVec2(1, 1));
        ImGui::End();

        ImGui::Begin("Inspector");
        auto objects = scene->FilterObjectsUnsafe([](auto) { return true; });
        m_Data->m_RegisteredGameObjects.clear();
        m_Data->m_RegisteredGameObjectsUUID.clear();
        for (auto& obj : objects)
        {
            auto UUID = obj->GetUUID();
            m_Data->m_RegisteredGameObjects.push_back(obj->GetName());
            m_Data->m_RegisteredGameObjectsUUID.push_back(UUID);
            if (UUID == m_Data->m_ActiveGameObjectUUID)
            {
                ImGui::Text("%s", obj->GetName().c_str());
                ImGui::TextDisabled("UUID: %s", obj->GetUUID().c_str());
                ImGui::Separator();
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
                        ImGui::PushID(component->GetUuid().c_str());
                        auto maxRows   = component->GetNumProperties();
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

                        ImGui::PopID();
                    }
                }
            }
        }
        ImGui::End();

        ImGui::Begin("Scene Hierarchy");
        RenderGameObjectTreeView(m_Data, scene);
        ImGui::End();

        ImGui::Begin("Console");
        PrintingLogs();
        ImGui::End();
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
