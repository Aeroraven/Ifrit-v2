## Ifrit-v2/TODO

### 4.0 Ongoing Schedule

- Subproject Ayanami:
  - Indirect Lighting on Surface Atlas (Debugging Phase)
  - Denoising / ReSTIR
- Subproject Syaro:
  - Refactoring with RDGs
- Architectural
  - Shader Rewrite: `ifrit.shader.neo`
  - Multithreaded Rendering
- Bug Fixing
  - (vkgraphics) Dangling view identifiers after resource destruction


### 4.1 Migration of  Project Aria

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

### 4.2 Syaro Refactoring

The architecture for subproject Syaro seems to be a little messy, following plans are scheduled:

- Render-Graph-Driven Process
- Redundant Dynamic Uniform Buffer Removal
- Shader Variants

### 4.3 Interview Feedback 

Following features are scheduled, with suggestions given by summer internship interviewers. 

- Optimized RDGs
- Multithreaded Rendering
- Branch Optimizations for Shaders
  - Change of Shader Languages: `Slang` or `HLSL` 
- `ddx` for Syaro
- Fixed-Point Math For Soft Renderer
