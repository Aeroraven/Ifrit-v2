## Ifrit-v2 / Requirements

Following extensions are required

### Vulkan Extensions/Features

- **Core Feature 1.0**
  - `samplerAnisotropy`
  - `geometryShader`
  - `shaderFloat64`
  - `shaderInt64`
  - `fragmentStoresAndAtomics`
  - `vertexPipelineStoresAndAtomics`
- **VK_KHR_shader_draw_parameters** (Core Feature 1.1)
  - `shaderDrawParameters`
- **VK_KHR_timeline_semaphore** (Core Feature 1.2)
  - `timelineSemaphore`
- **VK_EXT_descriptor_indexing** (Core Feature 1.2)
  - `descriptorIndexing`
  - `descriptorBindingPartiallyBound`
  - `descriptorBindingSampledImageUpdateAfterBind`
  - `descriptorBindingStorageBufferUpdateAfterBind`
  - `descriptorBindingStorageImageUpdateAfterBind`
  - `descriptorBindingStorageTexelBufferUpdateAfterBind`
  - `descriptorBindingUniformBufferUpdateAfterBind`
  - `descriptorBindingUniformTexelBufferUpdateAfterBind`
  - `descriptorBindingUpdateUnusedWhilePending`
  - `descriptorBindingVariableDescriptorCount`
  - `runtimeDescriptorArray`
- **VK_EXT_host_query_reset **(Core Feature 1.2)
  - `hostQueryReset`

- **VK_EXT_shader_image_atomic_int64 **(Core Feature 1.2)
  - `shaderImageInt64Atomics`
  - `shaderBufferInt64Atomics`
- **VK_KHR_shader_float16_int8 **(Core Feature 1.2)
  - `shaderFloat16`
- **VK_KHR_buffer_device_address **(Core Feature 1.2)
  - `bufferDeviceAddress`
- **VK_EXT_vertex_input_dynamic_state**
  - `vertexInputDynamicState`
- **VK_EXT_extended_dynamic_state3**
  - `extendedDynamicState3ColorBlendEnable`
  - `extendedDynamicState3LogicOpEnable`
  - `extendedDynamicState3ColorBlendEquation`
  - `extendedDynamicState3ColorWriteMask`
- **VK_EXT_extended_dynamic_state2**
  - `extendedDynamicState2`
  - `extendedDynamicState2LogicOp`
- **VK_EXT_extended_dynamic_state**
  - `extendedDynamicState`
- **VK_EXT_color_write_enable**
  - `colorWriteEnable`

- **VK_EXT_mesh_shader **
  - `taskShader`
  - `meshShader`