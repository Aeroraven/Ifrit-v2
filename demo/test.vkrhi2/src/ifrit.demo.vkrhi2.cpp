#ifndef IFRIT_DLL
    #define IFRIT_DLL
#endif

#define WINDOW_WIDTH 1500
#define WINDOW_HEIGHT 800

#include "ifrit/core/hal/HalDisplay.h"
#include "ifrit/core/hal/HalWindow.h"
#include "ifrit/display/presentation/window/WindowSelector.h"

#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/Queue.h"
#include "ifrit/vkrhi2/adapter/Backend.h"

#include <windows.h>
#include <iostream>

namespace Ifrit
{

} // namespace Ifrit

int main()
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

    Owner<RHI::RhiBackend> backend = MakeOwner<RHI::VulkanRHI2::VA_Backend>();
    backend->Init(initArgs);

    // Test Acquire
    auto castedBackend = static_cast<RHI::VulkanRHI2::VA_Backend*>(backend.get());
    auto device        = castedBackend->GetDevice();

    auto queues  = device->GetActiveQueues();
    auto cmdPool = queues.mGraphics->AcquireCommandPool();
    queues.mGraphics->ReleaseCommandPool(cmdPool);

    provider->Loop([](int* unused) {});

    return 0;
}
