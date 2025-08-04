## Ifrit-v2/TODO

### Ongoing Schedule

#### High Priority

- Subproject Artemis: `ifrit.runtime/physics.artemis`
- Architectural:
  - Serialization/Reflection
    - Stability Improvements

#### Normal Priority

- Subproject Artemis: `ifrit.runtime/physics.artemis`
  - Procedural Mesh Support

- Subproject Ayanami: `ifrit.runtime/render.ayanami`
  - Indirect Lighting on Surface Atlas (Debugging Phase)

  - Hardware Path Tracer (Reference Purpose)

  - Denoising / ReSTIR
- Subproject Syaro: `ifrit.runtime/render.syaro.v2`
  - Refactoring with RDGs
- RHI
  - HWRT Support

- Architectural
  - Dropping Legacy Designs:
    - Render Graph in `ifrit.vkgraphics`

  - Shader System:
    - Shader Rewrite In `slang/hlsl`: `ifrit.shader.neo`
    - Better Shader Cache System
  - Multithreaded Rendering
    - Async Compute
  - Streaming
- Bug Fixing
  - (vkgraphics) Dangling view identifiers after resource destruction
  - <s>(runtime/asset) Potential crash when loading gltf models</s> 
  - (shader/neo) Bindless declarations might violate `spirv-val` (but it does work)



### Migration of  Project Aria

Following features implemented in Aria/Vulkan might be considered to move into this repository:

- Hardware Ray Tracing
- NPR Shading
  - Outline (Post-Processing / Back Facing)
  - Rim Lighting
- Post Processing
  - FXAA
  - Kawase Blur
  - Global Fog
  - SSAO
  - <s>SSGI/SSR</s> 
    - SSGI-like tracing has been covered in `Ifrit.Runtime/Ayanami/ScreenProbeTracing`
- Volumetric Lighting
- Procedural Generator
  - GPU Marching Cubes

### Syaro Refactoring

The architecture for subproject Syaro seems to be a little messy, following plans are scheduled:

- Render-Graph-Driven Process
- Redundant Dynamic Uniform Buffer Removal
- Shader Variants

### Interview Feedback 

Following features are scheduled, with suggestions given by summer internship interviewers. 

- Optimized RDGs
- Multithreaded Rendering
- Branch Optimizations for Shaders
  - Change of Shader Languages: `Slang` or `HLSL` 
- `ddx` for Syaro
- Fixed-Point Math For Soft Renderer
