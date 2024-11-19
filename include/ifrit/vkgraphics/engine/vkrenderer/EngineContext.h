#pragma once
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <cstdint>
#include <vector>
#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <functional>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

struct IFRIT_APIDECL DeviceQueueInfo {
  struct DeviceQueueFamily {
    uint32_t m_familyIndex;
    uint32_t m_queueCount;
    VkFlags m_capability;
  };
  struct DeviceQueue {
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

struct IFRIT_APIDECL ExtensionFunction {
  PFN_vkCmdSetDepthTestEnable p_vkCmdSetDepthTestEnable;
  PFN_vkCmdSetDepthWriteEnable p_vkCmdSetDepthWriteEnable;
  PFN_vkCmdSetDepthCompareOp p_vkCmdSetDepthCompareOp;
  PFN_vkCmdSetDepthBoundsTestEnable p_vkCmdSetDepthBoundsTestEnable;
  PFN_vkCmdSetStencilTestEnable p_vkCmdSetStencilTestEnable;
  PFN_vkCmdSetStencilOp p_vkCmdSetStencilOp;

  PFN_vkCmdSetColorBlendEnableEXT p_vkCmdSetColorBlendEnableEXT;
  PFN_vkCmdSetColorWriteEnableEXT p_vkCmdSetColorWriteEnableEXT;
  PFN_vkCmdSetColorWriteMaskEXT p_vkCmdSetColorWriteMaskEXT;
  PFN_vkCmdSetColorBlendEquationEXT p_vkCmdSetColorBlendEquationEXT;
  PFN_vkCmdSetLogicOpEXT p_vkCmdSetLogicOpEXT;
  PFN_vkCmdSetLogicOpEnableEXT p_vkCmdSetLogicOpEnableEXT;
  PFN_vkCmdSetVertexInputEXT p_vkCmdSetVertexInputEXT;

  PFN_vkCmdDrawMeshTasksEXT p_vkCmdDrawMeshTasksEXT;
  PFN_vkCmdDrawMeshTasksIndirectEXT p_vkCmdDrawMeshTasksIndirectEXT;

  PFN_vkCmdBeginDebugUtilsLabelEXT p_vkCmdBeginDebugUtilsLabelEXT;
  PFN_vkCmdEndDebugUtilsLabelEXT p_vkCmdEndDebugUtilsLabelEXT;
};

class IFRIT_APIDECL EngineContext : public Rhi::RhiDevice {
private:
  constexpr static const char *s_validationLayerName =
      "VK_LAYER_KHRONOS_validation";
  Rhi::RhiInitializeArguments m_args;
  VkInstance m_instance;
  VkDebugUtilsMessengerEXT m_debugMessenger;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  DeviceQueueInfo m_queueInfo;
  VkDevice m_device;
  std::vector<const char *> m_instanceExtension = {
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
  std::vector<const char *> m_deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
      VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
      VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
      VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME,
      VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
      VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
      VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
      VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME,
      VK_KHR_SPIRV_1_4_EXTENSION_NAME,
      VK_EXT_MESH_SHADER_EXTENSION_NAME,
      VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME};
  VmaAllocator m_allocator;
  ExtensionFunction m_extf;
  VkPhysicalDeviceProperties m_phyDeviceProperties{};

private:
  void init();
  void loadExtensionFunction();
  void destructor();

public:
  EngineContext(const Rhi::RhiInitializeArguments &args);
  ~EngineContext();

  EngineContext(const EngineContext &) = delete;
  EngineContext &operator=(const EngineContext &) = delete;
  EngineContext(EngineContext &&) = delete;

  inline VkInstance getInstance() const { return m_instance; }
  inline VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
  inline VkDevice getDevice() const { return m_device; }
  inline const DeviceQueueInfo &getQueueInfo() const { return m_queueInfo; }
  inline const Rhi::RhiInitializeArguments &getArgs() const { return m_args; }
  inline const std::vector<const char *> &getDeviceExtensions() const {
    return m_deviceExtensions;
  }
  inline const VmaAllocator &getAllocator() const { return m_allocator; }
  void waitIdle();
  inline const ExtensionFunction getExtensionFunction() const { return m_extf; }
  inline const VkPhysicalDeviceProperties &getPhysicalDeviceProperties() const {
    return m_phyDeviceProperties;
  }
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics