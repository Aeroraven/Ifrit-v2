#ifndef IFRIT_DLL
    #define IFRIT_DLL
#endif

#define WINDOW_WIDTH 1500
#define WINDOW_HEIGHT 800

#include "ifrit/core/hal/HalDisplay.h"
#include "ifrit/core/hal/HalWindow.h"
#include "ifrit/core/hal/HalHostConcurrency.h"
#include "ifrit/display/presentation/window/WindowSelector.h"

#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/Queue.h"
#include "ifrit/vkrhi2/adapter/Backend.h"
#include "ifrit/vkrhi2/adapter/DisplayViewport.h"
#include "ifrit/core/algo/BuddyAllocator.h"
#include "ifrit/rhi/common/RhiShaderResource.h"
#include "ifrit/rhi/common/RhiDynamicUtils.h"
#include "ifrit/shadercompile/helper/ShaderCompileHelper.h"

#include <windows.h>
#include <iostream>

namespace Ifrit
{
    void testShaderRefl()
    {
        HAL::SetCurrentThreadId(0);
        ShaderCompile::ShaderCompileHelper helper;
        helper.SetIncludeBase("D:/Project2/Ifrit-v2/include");
        helper.SetCacheDir("D:/Project2/Ifrit-v2/.cache");
        auto output = helper.CompileShaderFromFile(
            "D:/Project2/Ifrit-v2/include/ifrit.shader.neo/Ayanami/Ayanami.Debug.GlobalDFRayMarch.comp.slang",
            "DebugGlobalDFRayMarchCS", {}, ShaderCompile::EShaderIRFormat::SpirV);
    }

    void testAllocator()
    {
        BuddyAddressAllocator allocator(4, 1024);

        auto                  res = allocator.Allocate(20);
        std::cout << "Allocated offset: " << res.mOffset << ", success: " << res.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;
        // allocator.Free(res.mOffset);
        // std::cout << "Freed offset: " << res.mOffset << std::endl;

        auto res2 = allocator.Allocate(16);
        std::cout << "Allocated offset: " << res2.mOffset << ", success: " << res2.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;
        // allocator.Free(res.mOffset);
        // std::cout << "Freed offset: " << res.mOffset << std::endl;

        auto res3 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res3.mOffset << ", success: " << res3.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res4 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res4.mOffset << ", success: " << res4.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res5 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res5.mOffset << ", success: " << res5.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res6 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res6.mOffset << ", success: " << res6.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res7 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res7.mOffset << ", success: " << res7.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res8 = allocator.Allocate(1024);
        std::cout << "Allocated offset: " << res8.mOffset << ", success: " << res8.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;
        return;
    }

    void testRhi2()
    {
        using namespace Ifrit;

        Owner<Display::Window::WindowProvider> provider;

        auto                                   dpiScaler = 1.0f;
        dpiScaler                                        = Ifrit::HAL::GetDisplayScale();

        // Setup Window
        Display::Window::WindowProviderSetupArgs winArgs;
        winArgs.useVulkan = true;
        Display::Window::WindowSelector     selector;
        Display::Window::WindowProviderType providerType;
        providerType = Display::Window::WindowProviderType::GLFW;

        provider = selector.CreateWindowProvider(providerType, winArgs);
        provider->Setup(static_cast<usize>(WINDOW_WIDTH), static_cast<usize>(WINDOW_HEIGHT));

        RHI::RhiInitializeArguments initArgs;
        initArgs.mSurfaceWidth      = WINDOW_WIDTH;
        initArgs.mSurfaceHeight     = WINDOW_HEIGHT;
        initArgs.mWin32.m_hInstance = GetModuleHandle(NULL);
        initArgs.mWin32.m_hWnd      = (HWND)provider->GetWindowObject();
        initArgs.mExtensionGetter   = [provider = provider.get()](u32* count) -> const char** {
            return provider->GetVkRequiredInstanceExtensions(count);
        };
        auto                   p         = new RHI::VulkanRHI2::VA_Backend();
        Owner<RHI::RhiBackend> vkBackend = MakeOwner<RHI::VulkanRHI2::VA_Backend>();
        RHI::SetRhiBackend(std::move(vkBackend));

        auto backend = RHI::GetRhiBackend();
        backend->Init(initArgs);

        auto taskScheduler = Task::GetTaskScheduler();

        auto task = taskScheduler->EnqueueTask(
            [&](Task::Task*, void*) {
                auto rhiShaderDesc        = RHI::RhiShaderCreateDesc();
                rhiShaderDesc.mEntryPoint = "DebugGlobalDFRayMarchCS";
                rhiShaderDesc.mSourceType = RHI::ERhiShaderSourceType::SlangCode;
                rhiShaderDesc.mStage      = RHI::ERhiShaderStage::Compute;
                rhiShaderDesc.mFilePath =
                    "C:/WR/Ifrit-v2/include/ifrit.shader.neo/Ayanami/Ayanami.Debug.GlobalDFRayMarch.comp.slang";
                rhiShaderDesc.mName = "DebugGlobalDFRayMarchCS";

                auto                             shader  = backend->CreateShader(rhiShaderDesc);
                auto                             variant = backend->GetShaderVariant("DebugGlobalDFRayMarchCS", {});

                RHI::RhiComputePipelineStateDesc compDesc;
                compDesc.mComputeShader = variant;
                auto pipeline           = backend->Experimental_GetComputePipeline(compDesc);
            },
            Task::ENamedTaskThread::AnyThread, {}, nullptr);
        taskScheduler->WaitForTask(task);

        provider->Loop([&](int* unused) {
            backend->BeginFrame();
            RHI::RhiTextureDesc desc = RHI::RhiTextureDesc::CreateTexture2D(512, 512, RHIPF_R32F, RHITexCreate_UAV, 1);
            auto                tex  = backend->CreateTexture(desc);
            auto                srv  = backend->CreateSRV(tex.get(), { 0, 0, 1, 1 });

            backend->EndFrame();
        });

        IF_LOG_INFO("IfritDemo", "RHI Backend released");
    }

} // namespace Ifrit

int main()
{
    // Ifrit::testShaderRefl();
    Ifrit::testRhi2();

    Ifrit::RHI::SetRhiBackend(nullptr);

    return 0;
}