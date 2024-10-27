#include <cstring>
#include <vector>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vkrenderer/include/utility/Logger.h>

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics {
bool enableExtension(
    bool mandatory, const char *extension,
    const std::vector<VkExtensionProperties> &availableExtensions,
    std::vector<const char *> &targetExtension) {
  for (auto ext : availableExtensions) {
    if (strcmp(ext.extensionName, extension) == 0) {
      targetExtension.push_back(extension);
      return true;
    }
  }
  if (mandatory) {

    // print all available extensions
    printf("Available extensions:\n");
    for (auto ext : availableExtensions) {
      vkrLog(ext.extensionName);
    }
    vkrError("Extension not found");
  }
  return false;
}
bool enableLayer(bool mandatory, const char *layer,
                 const std::vector<VkLayerProperties> &availableLayers,
                 std::vector<const char *> &targetLayers) {
  for (auto lay : availableLayers) {
    if (strcmp(lay.layerName, layer) == 0) {
      targetLayers.push_back(layer);
      return true;
    }
  }
  if (mandatory) {
    vkrError("Layer not found");
  }
  return false;
}
int physicalDeviceRanking(const VkPhysicalDevice &device) {
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(device, &properties);
  int score = 0;
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 100000;
  }
  score += properties.limits.maxImageDimension2D;
  return score;
}
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  vkrLog(pCallbackData->pMessage);
  std::abort();
  return VK_FALSE;
}

// START CLASS DEFINITION
template <typename T>
void loadExtFunc(T &extf, const char *name, VkDevice device) {
  extf = (T)vkGetDeviceProcAddr(device, name);
  if (!extf) {
    vkrError("Failed to load extension function");
  }
}

IFRIT_APIDECL void EngineContext::loadExtensionFunction() {
  loadExtFunc(m_extf.p_vkCmdSetDepthTestEnable, "vkCmdSetDepthTestEnable",
              m_device);
  loadExtFunc(m_extf.p_vkCmdSetDepthWriteEnable, "vkCmdSetDepthWriteEnable",
              m_device);
  loadExtFunc(m_extf.p_vkCmdSetDepthCompareOp, "vkCmdSetDepthCompareOp",
              m_device);
  loadExtFunc(m_extf.p_vkCmdSetDepthBoundsTestEnable,
              "vkCmdSetDepthBoundsTestEnable", m_device);
  loadExtFunc(m_extf.p_vkCmdSetStencilTestEnable, "vkCmdSetStencilTestEnable",
              m_device);
  loadExtFunc(m_extf.p_vkCmdSetStencilOp, "vkCmdSetStencilOp", m_device);

  loadExtFunc(m_extf.p_vkCmdSetColorBlendEnableEXT,
              "vkCmdSetColorBlendEnableEXT", m_device);
  loadExtFunc(m_extf.p_vkCmdSetColorWriteEnableEXT,
              "vkCmdSetColorWriteEnableEXT", m_device);
  loadExtFunc(m_extf.p_vkCmdSetColorWriteMaskEXT, "vkCmdSetColorWriteMaskEXT",
              m_device);
  loadExtFunc(m_extf.p_vkCmdSetColorBlendEquationEXT,
              "vkCmdSetColorBlendEquationEXT", m_device);
  loadExtFunc(m_extf.p_vkCmdSetLogicOpEXT, "vkCmdSetLogicOpEXT", m_device);
  loadExtFunc(m_extf.p_vkCmdSetLogicOpEnableEXT, "vkCmdSetLogicOpEnableEXT",
              m_device);
  loadExtFunc(m_extf.p_vkCmdSetVertexInputEXT, "vkCmdSetVertexInputEXT",
              m_device);
  loadExtFunc(m_extf.p_vkCmdDrawMeshTasksEXT, "vkCmdDrawMeshTasksEXT",
              m_device);
  loadExtFunc(m_extf.p_vkCmdDrawMeshTasksIndirectEXT,
              "vkCmdDrawMeshTasksIndirectEXT", m_device);

  vkrLog("Extension functions loaded");
}
IFRIT_APIDECL EngineContext::EngineContext(const InitializeArguments &args)
    : m_args(args) {
  init();
}
IFRIT_APIDECL void EngineContext::waitIdle() { vkDeviceWaitIdle(m_device); }
IFRIT_APIDECL void EngineContext::init() {
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "Ifrit-v2";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_3;

  VkInstanceCreateInfo instanceCI = {};
  instanceCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCI.pApplicationInfo = &appInfo;
  instanceCI.enabledExtensionCount = 0;
  instanceCI.ppEnabledExtensionNames = nullptr;
  instanceCI.enabledLayerCount = 0;
  instanceCI.ppEnabledLayerNames = nullptr;

  // Instance : Extensions
  uint32_t extensionCount = 0;
  vkrVulkanAssert(
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr),
      "Failed to enumerate instance extensions");
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkrVulkanAssert(vkEnumerateInstanceExtensionProperties(
                      nullptr, &extensionCount, availableExtensions.data()),
                  "Failed to enumerate instance extensions");
  std::vector<const char *> targetExtensions;

  if (m_args.m_extensionGetter) {
    uint32_t extensionCountExtra = 0;
    const char **extensionsExtra =
        m_args.m_extensionGetter(&extensionCountExtra);
    for (uint32_t i = 0; i < extensionCountExtra; i++) {
      enableExtension(true, extensionsExtra[i], availableExtensions,
                      targetExtensions);
    }
  }

  if (m_args.m_enableValidationLayer) {
    enableExtension(true, VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
                    availableExtensions, targetExtensions);
  }
  for (auto ext : m_instanceExtension) {
    enableExtension(true, ext, availableExtensions, targetExtensions);
  }

  instanceCI.enabledExtensionCount = targetExtensions.size();
  instanceCI.ppEnabledExtensionNames = targetExtensions.data();

  // Instance : Layers
  uint32_t layerCount = 0;
  vkrVulkanAssert(vkEnumerateInstanceLayerProperties(&layerCount, nullptr),
                  "Failed to enumerate instance layers");
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkrVulkanAssert(
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()),
      "Failed to enumerate instance layers");
  std::vector<const char *> targetLayers;

  if (m_args.m_enableValidationLayer) {
    enableLayer(true, s_validationLayerName, availableLayers, targetLayers);
  }
  instanceCI.enabledLayerCount = targetLayers.size();
  instanceCI.ppEnabledLayerNames = targetLayers.data();

  vkrVulkanAssert(vkCreateInstance(&instanceCI, nullptr, &m_instance),
                  "Failed to create Vulkan instance");
  vkrLog("Vulkan instance created");

  // Debug Messenger
  if (m_args.m_enableValidationLayer) {
    VkDebugUtilsMessengerCreateInfoEXT debugCI = {};
    debugCI.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCI.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCI.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCI.pfnUserCallback = debugCallback;
    debugCI.pUserData = nullptr;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (func) {
      vkrVulkanAssert(func(m_instance, &debugCI, nullptr, &m_debugMessenger),
                      "Failed to create debug messenger");
    } else {
      vkrError("Failed to create debug messenger");
    }
  }

  // Physical Device
  uint32_t physicalDeviceCount = 0;
  vkrVulkanAssert(
      vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr),
      "Failed to enumerate physical devices");
  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  vkrVulkanAssert(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount,
                                             physicalDevices.data()),
                  "Failed to enumerate physical devices");

  VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
  int bestScore = 0;
  for (auto device : physicalDevices) {
    int score = physicalDeviceRanking(device);
    if (score > bestScore) {
      bestScore = score;
      bestDevice = device;
    }
  }
  if (bestDevice == VK_NULL_HANDLE) {
    vkrError("No suitable physical device found");
  }
  m_physicalDevice = bestDevice;
  vkrLog("Physical device selected");

  // Physical Device Propertie
  vkGetPhysicalDeviceProperties(m_physicalDevice, &m_phyDeviceProperties);

  // Queue Family
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(bestDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(bestDevice, &queueFamilyCount,
                                           queueFamilies.data());
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    DeviceQueueInfo::DeviceQueueFamily family;
    family.m_familyIndex = i;
    family.m_queueCount = queueFamilies[i].queueCount;
    family.m_capability = queueFamilies[i].queueFlags;
    m_queueInfo.m_queueFamilies.push_back(family);
  }

  // Queue Creation
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::vector<std::vector<float>> queuePriorities;
  for (auto family : m_queueInfo.m_queueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = family.m_familyIndex;
    queueCreateInfo.queueCount = family.m_queueCount;
    queuePriorities.push_back(std::vector<float>(family.m_queueCount, 1.0f));
    queueCreateInfo.pQueuePriorities = queuePriorities.back().data();
    queueCreateInfos.push_back(queueCreateInfo);
  }

  // Device
  VkPhysicalDeviceFeatures deviceFeatures = {};
  VkPhysicalDeviceVulkan12Features deviceFeatures12 = {};
  VkPhysicalDeviceDynamicRenderingFeaturesKHR deviceFeaturesDynamic = {};
  VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT
      deviceFeaturesDynamicVertexInput = {};
  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT
      deviceFeaturesExtendedDynamicState3 = {};
  VkPhysicalDeviceExtendedDynamicState2FeaturesEXT
      deviceFeaturesExtendedState2 = {};
  VkPhysicalDeviceExtendedDynamicStateFeaturesEXT deviceFeaturesExtendedState =
      {};
  VkPhysicalDeviceColorWriteEnableFeaturesEXT deviceFeaturesColorWriteEnable =
      {};
  VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures{};
  VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{};

  deviceFeatures12.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  deviceFeatures12.timelineSemaphore = VK_TRUE;
  deviceFeatures12.descriptorIndexing = VK_TRUE;
  deviceFeatures12.descriptorBindingPartiallyBound = VK_TRUE;
  deviceFeatures12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
  deviceFeatures12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
  deviceFeatures12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
  deviceFeatures12.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
  deviceFeatures12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
  deviceFeatures12.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
  deviceFeatures12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
  deviceFeatures12.descriptorBindingVariableDescriptorCount = VK_TRUE;
  deviceFeatures12.runtimeDescriptorArray = VK_TRUE;
  deviceFeatures12.pNext = &deviceFeaturesDynamic;

  deviceFeaturesDynamic.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
  deviceFeaturesDynamic.dynamicRendering = VK_TRUE;
  deviceFeaturesDynamic.pNext = &deviceFeaturesDynamicVertexInput;

  deviceFeaturesDynamicVertexInput.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
  deviceFeaturesDynamicVertexInput.vertexInputDynamicState = VK_TRUE;
  deviceFeaturesDynamicVertexInput.pNext = &deviceFeaturesExtendedDynamicState3;

  deviceFeaturesExtendedDynamicState3.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
  deviceFeaturesExtendedDynamicState3.extendedDynamicState3ColorBlendEnable =
      VK_TRUE;
  deviceFeaturesExtendedDynamicState3.extendedDynamicState3LogicOpEnable =
      VK_TRUE;
  deviceFeaturesExtendedDynamicState3.extendedDynamicState3ColorBlendEquation =
      VK_TRUE;
  deviceFeaturesExtendedDynamicState3.extendedDynamicState3ColorWriteMask =
      VK_TRUE;
  deviceFeaturesExtendedDynamicState3.pNext = &deviceFeaturesExtendedState2;

  deviceFeaturesExtendedState2.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
  deviceFeaturesExtendedState2.extendedDynamicState2 = VK_TRUE;
  deviceFeaturesExtendedState2.extendedDynamicState2LogicOp = VK_TRUE;
  deviceFeaturesExtendedState2.pNext = &deviceFeaturesExtendedState;

  deviceFeaturesExtendedState.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
  deviceFeaturesExtendedState.extendedDynamicState = VK_TRUE;
  deviceFeaturesExtendedState.pNext = &deviceFeaturesColorWriteEnable;

  deviceFeaturesColorWriteEnable.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT;
  deviceFeaturesColorWriteEnable.colorWriteEnable = VK_TRUE;
  deviceFeaturesColorWriteEnable.pNext = &meshShaderFeatures;

  meshShaderFeatures.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
  meshShaderFeatures.taskShader = VK_TRUE;
  meshShaderFeatures.meshShader = VK_TRUE;

  deviceFeatures.samplerAnisotropy = VK_TRUE;

  VkDeviceCreateInfo deviceCI = {};
  deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCI.queueCreateInfoCount = queueCreateInfos.size();
  deviceCI.pQueueCreateInfos = queueCreateInfos.data();
  deviceCI.pEnabledFeatures = &deviceFeatures;
  deviceCI.pNext = &deviceFeatures12;

  // Device : Extensions
  std::vector<const char *> targetDeviceExtensions;
  uint32_t extensionCountDevice = 0;
  vkrVulkanAssert(vkEnumerateDeviceExtensionProperties(
                      bestDevice, nullptr, &extensionCountDevice, nullptr),
                  "Failed to enumerate device extensions");
  std::vector<VkExtensionProperties> availableExtensionsDevice(
      extensionCountDevice);
  vkrVulkanAssert(vkEnumerateDeviceExtensionProperties(
                      bestDevice, nullptr, &extensionCountDevice,
                      availableExtensionsDevice.data()),
                  "Failed to enumerate device extensions");
  for (auto extension : m_deviceExtensions) {
    enableExtension(true, extension, availableExtensionsDevice,
                    targetDeviceExtensions);
  }

  deviceCI.enabledExtensionCount = targetDeviceExtensions.size();
  deviceCI.ppEnabledExtensionNames = targetDeviceExtensions.data();

  // Device : Layers
  uint32_t layerCountDevice = 0;
  vkrVulkanAssert(
      vkEnumerateDeviceLayerProperties(bestDevice, &layerCountDevice, nullptr),
      "Failed to enumerate device layers");
  std::vector<VkLayerProperties> availableLayersDevice(layerCountDevice);
  vkrVulkanAssert(
      vkEnumerateDeviceLayerProperties(bestDevice, &layerCountDevice,
                                       availableLayersDevice.data()),
      "Failed to enumerate device layers");
  std::vector<const char *> targetLayersDevice;

  if (m_args.m_enableValidationLayer) {
    enableLayer(true, s_validationLayerName, availableLayersDevice,
                targetLayersDevice);
  }
  deviceCI.enabledLayerCount = targetLayersDevice.size();
  deviceCI.ppEnabledLayerNames = targetLayersDevice.data();
  vkrVulkanAssert(vkCreateDevice(bestDevice, &deviceCI, nullptr, &m_device),
                  "Failed to create logical device");
  vkrLog("Logical device created");

  // Retrieve Queues
  for (uint32_t i = 0; i < m_queueInfo.m_queueFamilies.size(); i++) {
    for (uint32_t j = 0; j < m_queueInfo.m_queueFamilies[i].m_queueCount; j++) {
      DeviceQueueInfo::DeviceQueue queue;
      vkGetDeviceQueue(m_device, i, j, &queue.m_queue);
      queue.m_familyIndex = i;
      queue.m_queueIndex = j;
      m_queueInfo.m_allQueues.push_back(queue);
      if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        m_queueInfo.m_graphicsQueues.push_back(queue);
      }
      if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        m_queueInfo.m_computeQueues.push_back(queue);
      }
      if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
        m_queueInfo.m_transferQueues.push_back(queue);
      }
    }
  }

  // Allocator
  VmaAllocatorCreateInfo allocatorCI = {};
  allocatorCI.vulkanApiVersion = VK_API_VERSION_1_3;
  allocatorCI.physicalDevice = m_physicalDevice;
  allocatorCI.device = m_device;
  allocatorCI.instance = m_instance;
  vmaCreateAllocator(&allocatorCI, &m_allocator);

  loadExtensionFunction();
  vkrLog("Allocator created");
}

void EngineContext::destructor() {
  vmaDestroyAllocator(m_allocator);
  vkDestroyDevice(m_device, nullptr);
  if (m_args.m_enableValidationLayer) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        m_instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func) {
      func(m_instance, m_debugMessenger, nullptr);
    }
  }
  vkDestroyInstance(m_instance, nullptr);
}

EngineContext::~EngineContext() { destructor(); }
} // namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics
