#pragma once
#include <common/core/ApiConv.h>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace Ifrit::Engine::VkRenderer{
    struct IFRIT_APIDECL InitializeArguments{
        const char**(*m_extensionGetter)(uint32_t* count) = nullptr;
        bool m_enableValidationLayer = true;
#ifdef _WIN32
        struct{
            HINSTANCE m_hInstance;
            HWND m_hWnd;
        } m_win32;
#else
        struct{
            void* m_hInstance;
            void* m_hWnd;
        } m_win32;
#endif
    };
    struct IFRIT_APIDECL DeviceQueueInfo{
        struct DeviceQueueFamily{
            uint32_t m_familyIndex;
            uint32_t m_queueCount;
        };
        struct DeviceQueue{
            VkQueue m_queue;
            uint32_t m_familyIndex;
            uint32_t m_queueIndex;
        };
        std::vector<DeviceQueueFamily> m_queueFamilies;
        std::vector<DeviceQueue> m_graphicsQueues;
        std::vector<DeviceQueue> m_computeQueues;
        std::vector<DeviceQueue> m_transferQueues;
        std::vector<DeviceQueue> m_allQueues;
    };
    class IFRIT_APIDECL EngineContext{
    private:
        constexpr static const char* s_validationLayerName = "VK_LAYER_KHRONOS_validation";
        InitializeArguments m_args;
        VkInstance m_instance;
        VkDebugUtilsMessengerEXT m_debugMessenger;
        VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
        DeviceQueueInfo m_queueInfo;
        VkDevice m_device;
        
    private:
        void init();
        void destructor();

    public:
        EngineContext(const InitializeArguments& args);
        ~EngineContext();

        EngineContext(const EngineContext&) = delete;
        EngineContext& operator=(const EngineContext&) = delete;
        EngineContext(EngineContext&&) = delete;

        inline VkInstance getInstance() const { return m_instance; }
        inline VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
        inline VkDevice getDevice() const { return m_device; }
        inline const DeviceQueueInfo& getQueueInfo() const { return m_queueInfo; }
        inline const InitializeArguments& getArgs() const { return m_args; }
    };
}