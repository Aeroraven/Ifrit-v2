#include "ifrit/ui/imgui/ImGuiProvider.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#define IMGUI_API IFRIT_APIDECL_IMPORT

namespace Ifrit::UI
{
    IFRIT_APIDECL void ImGuiProvider::OnInitialize(Runtime::IApplication* app)
    {
        Super::OnInitialize(app);
        auto displayProvider = app->GetDisplay();
        auto windowObject    = displayProvider->GetGLFWWindow();

        auto projectProperty = app->GetProjectProperty();
        iAssertion(
            projectProperty.m_rhiType == Runtime::AppRhiType::Vulkan, "ImGuiProvider only supports Vulkan RHI type");
        iAssertion(projectProperty.m_displayProvider == Runtime::AppDisplayProvider::GLFW,
            "ImGuiProvider only supports GLFW display provider");

        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(reinterpret_cast<GLFWwindow*>(windowObject), true);

        //ImGui_ImplVulkan_InitInfo init_info = {};
        //init_info.Instance                  = _instance;
        //init_info.PhysicalDevice            = _chosenGPU;
        //init_info.Device                    = _device;
        //init_info.Queue                     = _graphicsQueue;
        //init_info.DescriptorPool            = imguiPool;
        //init_info.MinImageCount             = 3;
        //init_info.ImageCount                = 3;
        //init_info.MSAASamples               = VK_SAMPLE_COUNT_1_BIT;
    }

    IFRIT_APIDECL void ImGuiProvider::OnShutdown() { Super::OnShutdown(); }

    IFRIT_APIDECL void ImGuiProvider::OnFrameBegin() {}

    IFRIT_APIDECL void ImGuiProvider::OnFrameEnd() {}
} // namespace Ifrit::UI
