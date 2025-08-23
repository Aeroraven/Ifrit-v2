#pragma once
#include <vulkan/vulkan.h>
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/vkrhi2/adapter/DeviceProcs.h"

namespace Ifrit::RHI::VulkanRHI2
{

    enum VA_MandatoryFlags
    {
        VA_Optional  = 0,
        VA_Mandatory = 1
    };

    template <typename T>
        requires IConceptIsFunctionPointer<T>
    void ConditionalLoadProc(T& funcPtr, VkDevice device, const char* functionName, bool condition)
    {
        if (condition)
        {
            funcPtr = reinterpret_cast<T>(vkGetDeviceProcAddr(device, functionName));
            if (!funcPtr)
            {
                IF_LOG_CRITICAL("VA_Device", "Failed to load device function pointer: {}", functionName);
                std::abort();
            }
        }
        else
        {
            funcPtr = nullptr;
        }
    }

    template <u32 IsMandatory> bool CheckMandatory(bool condition, const char* featureName)
    {
        if (condition)
        {
            return true;
        }
        else
        {
            if constexpr (IsMandatory)
            {
                IF_LOG_CRITICAL("VA_Device", "Mandatory feature '{}' is not supported", featureName);
                std::abort();
            }
            else
            {
                IF_LOG_WARNING("VA_Device", "Optional feature '{}' is not supported", featureName);
                return false;
            }
        }
    }

    template <typename T, auto S> T GetDeviceFeatureSupport(VkPhysicalDevice device)
    {
        T features{};
        features.sType = S;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features;

        vkGetPhysicalDeviceFeatures2(device, &features2);
        return features;
    }
    VkPhysicalDeviceFeatures GetDeviceFeatureSupport(VkPhysicalDevice physicalDevice)
    {
        VkPhysicalDeviceFeatures supportedFeatures{};
        vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);
        return supportedFeatures;
    }

    bool IsExtensionSupported(
        const char* extension, const Vec<const char*>& availableExtensions, bool mandatory = false)
    {
        if (extension == nullptr)
        {
            return true;
        }
        for (auto ext : availableExtensions)
        {
            if (strcmp(ext, extension) == 0)
            {
                IF_LOG_INFO("VA_Device", "Extension supported: {}", extension);
                return true;
            }
        }
        if (mandatory)
        {
            IF_LOG_CRITICAL("VA_Device", "Extension not found: {}", extension);
            std::abort();
        }
        else
        {
            IF_LOG_WARNING("VA_Device", "Extension is not supported: {}", extension);
        }
        return false;
    }

    using VA_DeviceProcLoadRequest = Fn<void(VkDevice device, VA_DeviceProcs& procs)>;

    class VA_DeviceExtension
    {
    public:
        virtual ~VA_DeviceExtension() = default;
        virtual void        OnQueryExtensionSupport(VkPhysicalDevice device, Vec<const char*>& availableExtensions) = 0;
        virtual void        OnQueryExtensionCapabilities(VkPhysicalDevice device)                                   = 0;
        virtual void        OnSetupExtensionCapabilities(RhiCapabilityList& caps, RhiPropertyList& props)           = 0;
        virtual void        AddExtensionToChain(void*& pNextChain)                                                  = 0;
        virtual const char* GetExtensionName() const                                                                = 0;

        void                RegisterProcLoader(VA_DeviceProcLoadRequest req) { mProcLoadRequests.push_back(req); }
        void                LoadProcs(VkDevice device, VA_DeviceProcs& procs)
        {
            for (auto& req : mProcLoadRequests)
            {
                req(device, procs);
            }
        }

    protected:
        Vec<VA_DeviceProcLoadRequest> mProcLoadRequests;
    };

    static Vec<VA_DeviceExtension*> gDeviceExtensions;

    class VA_AutoDeviceExtension : public VA_DeviceExtension
    {
    public:
        VA_AutoDeviceExtension() { gDeviceExtensions.push_back(this); }
    };

    template <typename T> void AddPNext(void*& head, T& toAdd)
    {
        toAdd.pNext = static_cast<typename std::remove_pointer<T>::type*>(head);
        head        = &toAdd;
    }

    template <> void AddPNext<VkPhysicalDeviceFeatures>(void*& head, VkPhysicalDeviceFeatures& toAdd)
    {
        IF_LOG_ASSERTION("VA_Device", head == nullptr, "VkPhysicalDeviceFeatures must be the last in the pNext chain");
        head = &toAdd;
    }

#define DECLARE_EXTENSION(structType, sTypeName, extensionName, extMandatoryReq)                                 \
    void OnSetupExtensionCapabilities_VA_EXT_##structType(structType& mRequired, const structType& mCaps,        \
        RhiCapabilityList& rhiCaps, RhiPropertyList& rhiProps, VA_DeviceExtension* devExt);                      \
    class VA_EXT_##structType : public VA_AutoDeviceExtension                                                    \
    {                                                                                                            \
    protected:                                                                                                   \
        structType          mData{};                                                                             \
        structType          mFinal{};                                                                            \
        bool                mSupported = false;                                                                  \
        virtual const char* GetExtensionName() const override { return (mSupported ? extensionName : nullptr); } \
        void                AddExtensionToChain(void*& pNextChain) override                                      \
        {                                                                                                        \
            if (!mSupported)                                                                                     \
                return;                                                                                          \
            AddPNext(pNextChain, mFinal);                                                                        \
        }                                                                                                        \
        void OnQueryExtensionCapabilities(VkPhysicalDevice device) override                                      \
        {                                                                                                        \
            if (!mSupported)                                                                                     \
                return;                                                                                          \
            mData = GetDeviceFeatureSupport<structType, sTypeName>(device);                                      \
        }                                                                                                        \
        void OnSetupExtensionCapabilities(RhiCapabilityList& caps, RhiPropertyList& props) override              \
        {                                                                                                        \
            if (!mSupported)                                                                                     \
                return;                                                                                          \
            OnSetupExtensionCapabilities_VA_EXT_##structType(mFinal, mData, caps, props, this);                  \
        }                                                                                                        \
        void OnQueryExtensionSupport(VkPhysicalDevice device, Vec<const char*>& availableExtensions) override    \
        {                                                                                                        \
            mSupported = IsExtensionSupported(extensionName, availableExtensions, extMandatoryReq);              \
        }                                                                                                        \
                                                                                                                 \
    public:                                                                                                      \
        VA_EXT_##structType()                                                                                    \
        {                                                                                                        \
            mData.sType  = sTypeName;                                                                            \
            mFinal.sType = sTypeName;                                                                            \
        }                                                                                                        \
    };                                                                                                           \
    static VA_EXT_##structType sExtensionRequirement_VA_EXT_##structType;                                        \
    void OnSetupExtensionCapabilities_VA_EXT_##structType(structType& mRequired, const structType& mCaps,        \
        RhiCapabilityList& rhiCaps, RhiPropertyList& rhiProps, VA_DeviceExtension* devExt)

#define DECLARE_EXTENSION_BASE()                                                                              \
    void OnSetupExtensionCapabilities_VA_EXT_VkPhysicalDeviceFeatures(                                        \
        VkPhysicalDeviceFeatures& mRequired, const VkPhysicalDeviceFeatures& mCaps);                          \
    class VA_EXT_VkPhysicalDeviceFeatures : public VA_AutoDeviceExtension                                     \
    {                                                                                                         \
    public:                                                                                                   \
        VkPhysicalDeviceFeatures mData{};                                                                     \
        VkPhysicalDeviceFeatures mFinal{};                                                                    \
                                                                                                              \
    protected:                                                                                                \
        bool                mSupported = true;                                                                \
        virtual const char* GetExtensionName() const override { return nullptr; }                             \
        void                AddExtensionToChain(void*& pNextChain) override {}                                \
        void                OnQueryExtensionCapabilities(VkPhysicalDevice device) override                    \
        {                                                                                                     \
            mData = GetDeviceFeatureSupport(device);                                                          \
        }                                                                                                     \
        void OnSetupExtensionCapabilities(RhiCapabilityList& rhiCaps, RhiPropertyList& rhiProps) override     \
        {                                                                                                     \
            OnSetupExtensionCapabilities_VA_EXT_VkPhysicalDeviceFeatures(mFinal, mData, rhiCaps, rhiProps);   \
        }                                                                                                     \
        void OnQueryExtensionSupport(VkPhysicalDevice device, Vec<const char*>& availableExtensions) override \
        {                                                                                                     \
            mSupported = true;                                                                                \
        }                                                                                                     \
                                                                                                              \
    public:                                                                                                   \
        VA_EXT_VkPhysicalDeviceFeatures() {}                                                                  \
    };                                                                                                        \
    static VA_EXT_VkPhysicalDeviceFeatures sExtensionRequirement_VA_EXT_VkPhysicalDeviceFeatures;             \
    void OnSetupExtensionCapabilities_VA_EXT_VkPhysicalDeviceFeatures(VkPhysicalDeviceFeatures& mRequired,    \
        const VkPhysicalDeviceFeatures& mCaps, RhiCapabilityList& rhiCaps, RhiPropertyList& rhiProps)

#define DECLARE_EXTENSION_NOFEATS(extensionName, extMandatoryReq)                                                  \
    void OnSetupExtensionCapabilities_VA_EXT_NOFEATS_##extensionName(                                              \
        RhiCapabilityList& rhiCaps, RhiPropertyList& rhiProps, VA_DeviceExtension* devExt);                        \
    class VA_EXT_NOFEATS_##extensionName : public VA_AutoDeviceExtension                                           \
    {                                                                                                              \
    protected:                                                                                                     \
        bool                mSupported = false;                                                                    \
        virtual const char* GetExtensionName() const override { return (mSupported ? #extensionName : nullptr); }  \
        void                AddExtensionToChain(void*& pNextChain) override {}                                     \
        void                OnQueryExtensionCapabilities(VkPhysicalDevice device) override {}                      \
        void                OnSetupExtensionCapabilities(RhiCapabilityList& caps, RhiPropertyList& props) override \
        {                                                                                                          \
            if (!mSupported)                                                                                       \
                return;                                                                                            \
            OnSetupExtensionCapabilities_VA_EXT_NOFEATS_##extensionName(caps, props, this);                        \
        }                                                                                                          \
        void OnQueryExtensionSupport(VkPhysicalDevice device, Vec<const char*>& availableExtensions) override      \
        {                                                                                                          \
            mSupported = IsExtensionSupported(extensionName, availableExtensions, extMandatoryReq);                \
        }                                                                                                          \
                                                                                                                   \
    public:                                                                                                        \
        VA_EXT_NOFEATS_##extensionName() {}                                                                        \
    };                                                                                                             \
    static VA_EXT_NOFEATS_##extensionName sExtensionRequirement_VA_EXT_NOFEATS_##extensionName;                    \
    void                                  OnSetupExtensionCapabilities_VA_EXT_NOFEATS_##extensionName(             \
        RhiCapabilityList& rhiCaps, RhiPropertyList& rhiProps, VA_DeviceExtension* devExt)

#define ENABLE_FEATURE(feature, mandatory) \
    mRequired.feature = CheckMandatory<VA_MandatoryFlags::mandatory>(mCaps.feature, #feature)

#define ENABLE_PROC_LOADER(procLoader, condition)                                        \
    devExt->RegisterProcLoader([=](VkDevice device, VA_DeviceProcs& procs) {             \
        ConditionalLoadProc(procs.p_##procLoader, device, #procLoader, mCaps.condition); \
    });

#define ENABLE_PROC_LOADER_ALWAYS(procLoader)                                 \
    devExt->RegisterProcLoader([=](VkDevice device, VA_DeviceProcs& procs) {  \
        ConditionalLoadProc(procs.p_##procLoader, device, #procLoader, true); \
    });

    // ===== Begin Extension Declarations =====

#define ALWAYS_SUPPORT nullptr

    DECLARE_EXTENSION_BASE()
    {
        ENABLE_FEATURE(samplerAnisotropy, VA_Mandatory);
        ENABLE_FEATURE(geometryShader, VA_Mandatory);
        ENABLE_FEATURE(shaderFloat64, VA_Mandatory);
        ENABLE_FEATURE(shaderInt64, VA_Mandatory);
        ENABLE_FEATURE(shaderInt16, VA_Mandatory);
        ENABLE_FEATURE(fragmentStoresAndAtomics, VA_Mandatory);
        ENABLE_FEATURE(vertexPipelineStoresAndAtomics, VA_Mandatory);
    }

    DECLARE_EXTENSION_NOFEATS(VK_KHR_SWAPCHAIN_EXTENSION_NAME, VA_Mandatory) {}
    DECLARE_EXTENSION_NOFEATS(VK_KHR_SPIRV_1_4_EXTENSION_NAME, VA_Mandatory) {}
    DECLARE_EXTENSION_NOFEATS(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME, VA_Optional)
    {
        rhiCaps.bConservativeRasterizationEnabled = true;
    }
    DECLARE_EXTENSION_NOFEATS(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VA_Optional)
    {
        ENABLE_PROC_LOADER_ALWAYS(vkCmdBeginDebugUtilsLabelEXT);
        ENABLE_PROC_LOADER_ALWAYS(vkCmdEndDebugUtilsLabelEXT);
        ENABLE_PROC_LOADER_ALWAYS(vkSetDebugUtilsObjectNameEXT);
        ENABLE_PROC_LOADER_ALWAYS(vkSetDebugUtilsObjectTagEXT);
    }

    DECLARE_EXTENSION(VkPhysicalDeviceVulkan11Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        ALWAYS_SUPPORT, VA_Mandatory)
    {
        ENABLE_FEATURE(shaderDrawParameters, VA_Mandatory);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        ALWAYS_SUPPORT, VA_Mandatory)
    {
        ENABLE_FEATURE(timelineSemaphore, VA_Mandatory);
        ENABLE_FEATURE(descriptorIndexing, VA_Mandatory);
        ENABLE_FEATURE(descriptorBindingPartiallyBound, VA_Mandatory);
        ENABLE_FEATURE(descriptorBindingSampledImageUpdateAfterBind, VA_Mandatory);
        ENABLE_FEATURE(descriptorBindingStorageBufferUpdateAfterBind, VA_Mandatory);
        ENABLE_FEATURE(descriptorBindingStorageImageUpdateAfterBind, VA_Mandatory);
        ENABLE_FEATURE(descriptorBindingStorageTexelBufferUpdateAfterBind, VA_Optional);
        ENABLE_FEATURE(descriptorBindingUniformBufferUpdateAfterBind, VA_Optional);
        ENABLE_FEATURE(descriptorBindingUniformTexelBufferUpdateAfterBind, VA_Optional);
        ENABLE_FEATURE(descriptorBindingUpdateUnusedWhilePending, VA_Mandatory);
        ENABLE_FEATURE(descriptorBindingVariableDescriptorCount, VA_Mandatory);
        ENABLE_FEATURE(runtimeDescriptorArray, VA_Mandatory);
        ENABLE_FEATURE(hostQueryReset, VA_Mandatory);
        ENABLE_FEATURE(shaderSharedInt64Atomics, VA_Mandatory);
        ENABLE_FEATURE(shaderBufferInt64Atomics, VA_Mandatory);
        ENABLE_FEATURE(shaderFloat16, VA_Optional);
        ENABLE_FEATURE(bufferDeviceAddress, VA_Mandatory);

        ENABLE_PROC_LOADER(vkGetBufferDeviceAddress, bufferDeviceAddress);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceDynamicRenderingFeaturesKHR,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VA_Mandatory)
    {
        ENABLE_FEATURE(dynamicRendering, VA_Mandatory);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
        VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME, VA_Mandatory)
    {
        ENABLE_FEATURE(vertexInputDynamicState, VA_Mandatory);

        ENABLE_PROC_LOADER(vkCmdSetVertexInputEXT, vertexInputDynamicState);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceExtendedDynamicState3FeaturesEXT,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, VA_Mandatory)
    {
        ENABLE_FEATURE(extendedDynamicState3ColorBlendEnable, VA_Mandatory);
        ENABLE_FEATURE(extendedDynamicState3LogicOpEnable, VA_Mandatory);
        ENABLE_FEATURE(extendedDynamicState3ColorBlendEquation, VA_Mandatory);
        ENABLE_FEATURE(extendedDynamicState3ColorWriteMask, VA_Mandatory);

        ENABLE_PROC_LOADER(vkCmdSetColorBlendEnableEXT, extendedDynamicState3ColorBlendEnable);
        ENABLE_PROC_LOADER(vkCmdSetLogicOpEnableEXT, extendedDynamicState3LogicOpEnable);
        ENABLE_PROC_LOADER(vkCmdSetColorBlendEquationEXT, extendedDynamicState3ColorBlendEquation);
        ENABLE_PROC_LOADER(vkCmdSetColorWriteMaskEXT, extendedDynamicState3ColorWriteMask);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceExtendedDynamicState2FeaturesEXT,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT,
        VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, VA_Mandatory)
    {
        ENABLE_FEATURE(extendedDynamicState2, VA_Mandatory);
        ENABLE_FEATURE(extendedDynamicState2LogicOp, VA_Mandatory);

        ENABLE_PROC_LOADER(vkCmdSetLogicOpEXT, extendedDynamicState2LogicOp);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceExtendedDynamicStateFeaturesEXT,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, VA_Mandatory)
    {
        ENABLE_FEATURE(extendedDynamicState, VA_Mandatory);

        ENABLE_PROC_LOADER(vkCmdSetDepthTestEnable, extendedDynamicState);
        ENABLE_PROC_LOADER(vkCmdSetDepthWriteEnable, extendedDynamicState);
        ENABLE_PROC_LOADER(vkCmdSetDepthCompareOp, extendedDynamicState);
        ENABLE_PROC_LOADER(vkCmdSetDepthBoundsTestEnable, extendedDynamicState);
        ENABLE_PROC_LOADER(vkCmdSetStencilTestEnable, extendedDynamicState);
        ENABLE_PROC_LOADER(vkCmdSetStencilOp, extendedDynamicState);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceColorWriteEnableFeaturesEXT,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT, VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME,
        VA_Mandatory)
    {
        ENABLE_FEATURE(colorWriteEnable, VA_Mandatory);

        ENABLE_PROC_LOADER(vkCmdSetColorWriteEnableEXT, colorWriteEnable);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VA_Optional)
    {
        ENABLE_FEATURE(shaderBufferFloat32Atomics, VA_Optional);
        ENABLE_FEATURE(shaderBufferFloat32AtomicAdd, VA_Optional);
        ENABLE_FEATURE(shaderSharedFloat32AtomicAdd, VA_Optional);
        ENABLE_FEATURE(shaderSharedFloat32Atomics, VA_Optional);
    };

    // ===== Optional Extensions =====

    DECLARE_EXTENSION(VkPhysicalDeviceMeshShaderFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
        VK_EXT_MESH_SHADER_EXTENSION_NAME, VA_Optional)
    {
        rhiCaps.bMeshShaderEnabled = true;

        ENABLE_FEATURE(taskShader, VA_Optional);
        ENABLE_FEATURE(meshShader, VA_Optional);

        ENABLE_PROC_LOADER(vkCmdDrawMeshTasksEXT, meshShader);
        ENABLE_PROC_LOADER(vkCmdDrawMeshTasksIndirectEXT, meshShader);
    };

    DECLARE_EXTENSION(VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
        VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, VA_Optional)
    {
        ENABLE_FEATURE(shaderImageInt64Atomics, VA_Optional);
    };

    // ===== Helpers =====
    struct VA_DeviceExtensionData
    {
        void*                     mExtensionChain = nullptr;
        VkPhysicalDeviceFeatures* mBaseFeatures   = nullptr;
        Vec<const char*>          mEnabledExtensions;
    };

    VA_DeviceExtensionData PrepareDeviceExtension(VkPhysicalDevice device, Vec<const char*>& availableExtensions,
        RhiCapabilityList& rhiCaps, RhiPropertyList& rhiProps)
    {
        VA_DeviceExtensionData extData;
        for (auto ext : gDeviceExtensions)
        {
            ext->OnQueryExtensionSupport(device, availableExtensions);
        }
        Vec<const char*> enabledExtensions;
        for (auto ext : gDeviceExtensions)
        {
            if (const char* name = ext->GetExtensionName(); name != nullptr)
            {
                enabledExtensions.push_back(name);
            }
        }
        extData.mEnabledExtensions = std::move(enabledExtensions);

        for (auto ext : gDeviceExtensions)
        {
            ext->OnQueryExtensionCapabilities(device);
            ext->OnSetupExtensionCapabilities(rhiCaps, rhiProps);
            ext->AddExtensionToChain(extData.mExtensionChain);
        }

        auto firstExt = gDeviceExtensions.front();
        auto casted   = dynamic_cast<VA_EXT_VkPhysicalDeviceFeatures*>(firstExt);
        if (casted)
        {
            extData.mBaseFeatures = &casted->mFinal;
        }
        else
        {
            IF_LOG_CRITICAL("VA_Device", "First extension is not VkPhysicalDeviceFeatures");
            std::abort();
        }

        return extData;
    }

    VA_DeviceProcs LoadDeviceProcs(VkDevice device)
    {
        VA_DeviceProcs procs;
        for (auto ext : gDeviceExtensions)
        {
            ext->LoadProcs(device, procs);
        }
        return procs;
    }

} // namespace Ifrit::RHI::VulkanRHI2