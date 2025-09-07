## Ifrit-v2/TODO

### Ongoing Schedule

#### High Priority (Ongoing)

- Architectural:
  - **RHI Refactoring (Ongoing, from interview feedback)**
    - Command Translation Layer
    - Drop Queue Exposure
    - Better Shader Parameter Setup
    - Disentangle Views and Resources
    - Multithreading Support
  - Serialization/Reflection
    - Stability Improvements
  - **Multithreaded Rendering (Ongoing)**
    - Async Compute
    - Refactored RDG
    - Parallel Command Recording (RDG Support)
    - Refined Resource Upload
      - Resource Streaming
  - Cleanup
    - Drop Unused Submodules: `cereal`

#### Normal Priority

- Subproject Artemis: `ifrit.runtime/physics.artemis`
  - <s>Procedural Mesh Support</s>
- Subproject Ayanami: `ifrit.runtime/render.ayanami`
  - Indirect Lighting on Surface Atlas (Debugging Phase)

  - Hardware Path Tracer (Reference Purpose)

  - Denoising / ReSTIR
- Subproject Syaro: `ifrit.runtime/render.syaro.v2`
  - Refactoring with RDGs
- RHI
  - DX12 Support
  - HWRT Support

- Architectural
  - Dropping Legacy Designs:
    - Render Graph in `ifrit.vkgraphics`
  - Shader System:
    - Shader Rewrite In `slang/hlsl`: `ifrit.shader.neo`
    - Better Shader Cache System
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
  - <s>GPU Marching Cubes</s>
    - GPU Marching Cubes has been covered in `Ifrit.Runtime/Geometry/SurfaceOp` as a part of subproject `Artemis`

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
