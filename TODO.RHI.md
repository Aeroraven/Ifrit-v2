## Ifrit-v2/TODO

### Existing RHI Problems 

- Improper abstraction
  - Unnecessary concept exposure: like queues
- Improper design pattern adoption
  - Coupled design between viewable resources (buffer, texture) and views (UAV,SRV)
  - Improper ownership identification
- Hard to parallelizing CPU command recording, caused by
  - Per-resource state tracking 
  - Coupled design among command pool, queues and command lists
    - Lacking the separation between upload context and executing context
- Redundant pipeline state setup
  - Lacking of shader meta data extraction (like SPIRV headers) and shader reflection

### Goals

- Expose command list instead of queues
  - `CommandTask` for a platform-specific command list (`CommandBuffer` in Vulkan)
  - `CommandListContext` for a sequence of platform-specific command lists for a certain capacity (like async compute)
  - `CommandList` for a upper-level encapsulation: when command flushes, submit uploading context first then executing context.
    - Uploading context contains things that should be done immediately, like initial layout setting and staged data transferring.
    - Not thread-safe. Each worker should allocate a new cmd list.
- Redesign of pipeline state and shader classes
  - `Shader` class should carries more information like entry and descriptor binding layout info.
  - `Shader` should be class instead of just path
- Stateless resources
  - Only `initialLayout` is managed by RHI resources. Transition tracking should be done in upper-level RDGs
- Integration of existing task system

