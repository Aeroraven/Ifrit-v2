#include "ifrit/ui/imgui/ImGuiProvider.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/runtime/base/Property.h"
#include "ifrit/runtime/base/Scene.h"

#include "glfw/glfw3.h"

#define IMGUI_API IFRIT_APIDECL_IMPORT

namespace Ifrit::UI
{
    static VkFormat imguiColorAttachmentFormats[] = { VK_FORMAT_B8G8R8A8_SRGB };

    static void     imguiCheckResultFn(VkResult err)
    {
        if (err == VK_SUCCESS)
            return;
        iAssertion(false, "ImGui Vulkan error: {}", (int)err);
    }

    static void RegisterEditorHandles(ImGuiProvider* provider)
    {
        auto& f32Handles            = Runtime::GetPropertyEditorHandle<f32>();
        f32Handles.m_SliderCallback = [](const char* name, f32& value, f32 min, f32 max, f32 step) {
            ImGui::SliderFloat(name, &value, min, max, "%.3f", ImGuiSliderFlags_AlwaysClamp);
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

    IFRIT_APIDECL void ImGuiProvider::OnInitialize(Runtime::IApplication* app)
    {

        Super::OnInitialize(app);
        auto displayProvider = app->GetDisplay();
        auto windowObject    = displayProvider->GetGLFWWindow();

        auto projectProperty = app->GetProjectProperty();
        auto rhi             = app->GetRhi();
        auto dq              = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Graphics);
        iAssertion(
            projectProperty.m_rhiType == Runtime::AppRhiType::Vulkan, "ImGuiProvider only supports Vulkan RHI type");
        iAssertion(projectProperty.m_displayProvider == Runtime::AppDisplayProvider::GLFW,
            "ImGuiProvider only supports GLFW display provider");

        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        io.DisplaySize.x           = projectProperty.m_width;
        io.DisplaySize.y           = projectProperty.m_height;

        ImGui::StyleColorsDark();
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
        io.DisplaySize.x = m_Application->GetProjectProperty().m_width;
        io.DisplaySize.y = m_Application->GetProjectProperty().m_height;

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Goodbye, world!");

        ImGui::End();
        ImGui::Begin("Hello, world!");
        // ImGui::Text("This is some useful text.");
    }

    IFRIT_APIDECL void ImGuiProvider::OnFrameEnd() {}

    IFRIT_APIDECL void ImGuiProvider::OnUpdate(Runtime::Scene* scene)
    {
        auto objects = scene->FilterObjectsUnsafe([](auto) { return true; });
        for (auto& obj : objects)
        {
            ImGui::Text("GameObject: %s", obj->GetName().c_str());
            auto components = obj->GetAllComponents();
            for (auto& component : components)
            {
                component->CallPropertyEditorHandle();
            }
        }
    }

    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> ImGuiProvider::OnPostRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        ImGui::End();
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
            [draw_data, swapchainView](const RHI::RhiCommandList* cmd) {
                cmd->BeginScope("UI: ImGui Rendering");
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

                vkCmdBeginRendering(vkcmd, &renderingInfo);
                if (draw_data->Valid)
                {
                    ImGui_ImplVulkan_RenderDrawData(draw_data, vkcmd);
                }
                else
                {
                    iWarn("ImGui draw data is not valid, skipping rendering.");
                }
                vkCmdEndRendering(vkcmd);
                cmd->EndScope();
            },
            { prevSubmission }, {});
    }
} // namespace Ifrit::UI
