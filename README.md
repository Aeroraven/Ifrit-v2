# Ifrit-v2

GPU/CPU-Parallelized tile-based software rasterizer.

![](./img/img_demo0png.png)

![](img/img_demo3.png)

![](img/img_demo1.png)

![](img/img_demo2.png)



Check  **[`gallery.md`](./docs/gallery.md)** for **[More Example Pictures](./gallery.md)**



Successor to following repos:

 - [Ifrit](https://github.com/Aeroraven/Ifrit)
 - [Iris (TinyRenderer CPP)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris)
 - [Iris (TinyRenderer C#)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/TinyRenderer)





## Features

Overall framework for CUDA solid triangle renderer pipeline (Some are different from its MT-CPU counterpart). Stages with asterisk mark are optional. Tiling optimization is only applied for filled triangles.

<img src="./img/overview.png" alt="overview" style="zoom: 67%;" />

**Note:** This project is NOT an exact replicate of hardware graphics pipeline (like IMR or TBDR architecture). 

✅ Available | 🟦 Limited  Support | 🟥 TODO

| Feature                                     | [Iris Renderer](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris) | MT CPU Renderer | CUDA Renderer |
| ------------------------------------------- | ------------------------------------------------------------ | --------------- | ------------- |
|                                             |                                                              |                 |               |
| 🔗 **Integration (Wrapper)**                 |                                                              |                 |               |
| C++ DLL                                     | 🟥                                                            | 🟦               | 🟥             |
| .NET Library (`C#`)                         | 🟥                                                            | 🟦               | 🟥             |
| 🔗 **Shader Language**                       |                                                              |                 |               |
| In-Application Class                        | ✅                                                            | ✅               | ✅             |
| SPIR-V Binary / HLSL                        | 🟥                                                            | 🟦 OrcJIT (2)    | 🟥             |
| SPIR-V Binary / GLSL                        | 🟥                                                            | 🟥               | 🟥             |
| 🚀 **Ray-tracer / Performance**              |                                                              |                 |               |
| SIMD Instructions / SIMT                    | 🟥                                                            | ✅               | 🟥             |
| Acceleration Structure (BVH)                | 🟥                                                            | ✅               | 🟥             |
| Lock-free Synchronization                   | 🟥                                                            | ✅               | 🟥             |
| 🔦 **Ray-tracer / Pipeline**                 |                                                              |                 |               |
| Acceleration Structure Traversal            | 🟥                                                            | 🟦               | 🟥             |
| Surface Area Heuristic                      | 🟥                                                            | ✅               | 🟥             |
| Programmable Ray Generation Shader          | 🟥                                                            | ✅               | 🟥             |
| Programmable Closest Hit Shader             | 🟥                                                            | ✅               | 🟥             |
| Programmable Miss Shader                    | 🟥                                                            | ✅               | 🟥             |
| 🚀 **Rasterization / Performance**           |                                                              |                 |               |
| SIMD Instructions / SIMT                    | 🟥                                                            | ✅               | ✅             |
| Overlapped Memory Transfer                  | 🟥                                                            | 🟥               | ✅             |
| Dynamic Tile List                           | 🟥                                                            | ✅               | ✅             |
| Lock-free Synchronization                   | 🟥                                                            | ✅               | ⬜             |
| 💡 **Rasterization / Basic**                 |                                                              |                 |               |
| Rendering Order                             | ✅                                                            | ✅               | ✅             |
| 💡 **Rasterization / Pipeline**              |                                                              |                 |               |
| Programmable Vertex Shader                  | ✅                                                            | ✅               | ✅             |
| Programmable Pixel Shader                   | ✅                                                            | ✅               | ✅             |
| Programmable Geometry Shader                | 🟥                                                            | 🟥               | 🟦             |
| Programmable Mesh Shader                    | 🟥                                                            | 🟥               | 🟦             |
| Programmable Task Shader                    | 🟥                                                            | 🟥               | 🟦             |
| Alpha Blending                              | 🟥                                                            | ✅               | ✅             |
| Depth Testing                               | ✅                                                            | ✅               | ✅             |
| Depth Function                              | 🟥                                                            | ✅               | ✅             |
| Z Pre-Pass                                  | 🟥                                                            | ✅               | ✅             |
| Early-Z Test                                | ✅                                                            | ✅               | ✅             |
| Late-Z Test (Depth Replacement & `discard`) | 🟥                                                            | 🟥               | ✅             |
| Scissor Test                                | 🟥                                                            | 🟥               | ✅             |
| Back Face Culling                           | ✅                                                            | ✅               | ✅             |
| Frustum Culling                             | 🟥                                                            | ✅               | ✅             |
| Homogeneous Clipping                        | 🟥                                                            | ✅               | ✅             |
| Small Triangle Culling                      | 🟥                                                            | ✅               | ✅             |
| Perspective-correct Interpolation           | 🟥                                                            | ✅               | ✅             |
| Shader Derivatives `dFdx` `dFdy`            | 🟥                                                            | 🟥               | 🟦             |
| Multi-sampling                              | 🟥                                                            | 🟥               | 🟦 8x MSAA     |
| 💡 **Rasterization / Polygon Mode**          |                                                              |                 |               |
| Filled Triangle                             | ✅                                                            | ✅               | ✅             |
| Line (Wireframe)                            | 🟥                                                            | 🟥               | 🟦             |
| Point                                       | 🟥                                                            | 🟥               | 🟦             |
| 🖼️ **Texture**                               |                                                              |                 |               |
| Basic Support (Sampler)                     | 🟥                                                            | 🟥               | ✅             |
| Blit                                        | 🟥                                                            | 🟥               | ✅             |
| Mipmap                                      | 🟥                                                            | 🟥               | ✅             |
| Filter                                      | 🟥                                                            | 🟥               | ✅             |
| Sampler Address Mode                        | 🟥                                                            | 🟥               | ✅             |
| LOD Bias                                    | 🟥                                                            | 🟥               | ✅             |
| Anisotropic Filtering                       | 🟥                                                            | 🟥               | 🟦             |
| Cube Map                                    | 🟥                                                            | 🟥               | ✅             |
| 🖥️ **Presentation**                          |                                                              |                 |               |
| Terminal ASCII                              | 🟥                                                            | ✅               | ✅             |
| Terminal Color                              | 🟥                                                            | ✅               | ✅             |

(1) Shader derivatives are now only available for the filled triangle polygon mode. Shader derivatives are calculated in `2x2` quads, so precision might matter.

(2) Partial instructions are supported. Only available for binaries produced by `glslc` or `dxc`

### Supported Feature Details

- Sampler Filter :`IF_FILTER_NEAREST`, `IF_FILTER_LINEAR`
- Sampler Address Mode: `IF_SAMPLER_ADDRESS_MODE_REPEAT`, `IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE`, `IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER` , `IF_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT`, `IF_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE`



## Performance

### Frame Rate Comparison (FPS)  Version 2

#### Influence of Attachment Size

Tests performed on multi-thread CPU renderer (1 master + 16 workers), with just-in-time(JIT) compilation of Vulkan-specific HLSL shaders (compiled in SPIR-V binary format). All attachments are in `linear` tiling mode and `float32` mode. 

| Model                             | 512 x 512 | 1024 x 1024 | 2048 x 2048 | 4096 x 4096 |
| --------------------------------- | --------- | ----------- | ----------- | ----------- |
| Kirara / Genshin Impact (3.7 k)   | 1219      | 480         | 124         | 28          |
| Evil Neurosama (55.9 k)           | 606       | 398         | 120         | 31          |
| Flandre Scarlet / Touhou (96.1 k) | 502       | 237         | 82          | 19          |
| Miyako / Blue Archive (346.1 k)   | 106       | 72          | 43          | 13          |



#### Influence of Triangle Numbers

Tests performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored. Note that some triangles **might be culled or clipped** in the pipeline. 

| Model                                   | Yomiya     | Bunny      | Khronos Sponza | Intel Sponza |
| --------------------------------------- | ---------- | ---------- | -------------- | ------------ |
| Triangles                               | 70275      | 208353     | 786801         | 11241912     |
| Single Thread CPU Baseline v1           | 38         | 20         | 2              | 1            |
| Multi Thread CPU Baseline v1            | 80         | 80         | 10             | 2            |
| CUDA Baseline v1                        | 2857       | 2272       | 500            | 198          |
| ST CPU Optimized v2 (C++ / SPIR-V HLSL) | 47 (+23%)  | 27 (+35%)  | 4 (+100%)      | 2 (+100%)    |
| MT CPU Optimized v2 (C++ / SPIR-V HLSL) | 134 (+68%) | 108 (+35%) | 24 (+140%)     | 11 (+450%)   |
| ST CPU Optimized v2 (C++ / Class)       | 50 (+31%)  | 29 (+45%)  | 4 (+100%)      | 2 (+100%)    |
| MT CPU Optimized v2 (C++ / Class)       | 138 (+73%) | 110 (+38%) | 25 (+150%)     | 15 (+650%)   |
| MT CPU Optimized v2 (C# / SPIR-V HLSL)  |            |            |                |              |

※ **C++ Class**: shaders are coded and compiled ahead-of-time, using virtual inheritance.

※ **SPIR-V HLSL (C++)**: all shader codes are compiled into binary form using `glslc`. HLSL source codes are written in `Vulkan-specific` style. Just-in-time (JIT) compilation uses LLVM 10 as backend and manual IR mapping (Shared library is compiled with `mingw-w64`). App runs in `msvc`.

※ **SPIR-V HLSL (C#)**: the same as above, but with .NET `P/Invoke Source Generation`(`LibraryImport`) that invokes shared library compiled using C++.



### **Frame Rate Comparison (FPS)  Version 1**

See [Performance History](./docs/performance.md)



### Test Environment

- CPU: 12th Gen Intel(R) Core(TM) i9-12900H 
  - Test with 17 threads (1 master + 16 workers) + AVX2 Instructions
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (Kernel parameters are optimized for the SPECIFIC test environment)
- Shading: World-space normal
- Compiler:  MSVC 19.29 + NVCC 12.6 (with O3 optimization)



## Dependencies

See [Requirements & Build Instructions ](./docs/requirement.md)for more details.



## Setup / Run

See [Requirements & Build Instructions ](./docs/requirement.md)for more details.



## Abstractions / Usage

See  [Usage](./docs/docs.md) for more details.



## Ongoing Plan

### Long-term Plan

- [ ] Tessellation
- [x] Line Mode
- [x] Texture LOD & Texture Sampler
  - [x] Shader Derivatives
  - [x] Anisotropic Filtering
  - [x] Dynamic LOD Selection & Texture Bias
  - [x] Cube Mapping
  - [ ] Tiling
- [x] Multi-sampling
  - [ ] Blending Integration
- [x] Alpha Blending
  - [x] Sorting
- [x] Mesh Shader
- [x] Shader Binary
  - [ ] Matrix Operations
  - [ ] Optimization
- [ ] Input Topology
- [ ] Triangle Cluster & Cluster LOD
- [ ] Known Issues
  - [x] Issue: Faults after resolution change
  - [ ] Overdraw: Point mode with index buffer
  - [ ] Latency: Excessive global atomics in line mode
  - [ ] Issue: Nondeterministic behaviors in wireframe/point mode 
  - [x] Issue: Artifacts in low resolution scenario 
  - [ ] Latency: Memory access pattern in MSAA
  - [ ] Latency: JIT slows down execution (in raytracer)
- [ ] Standardization
  - [ ] C++: `-Wignored-attributes` warnings in SIMD class  





## References

For models / open source code references, check `licenses` folder.

[1]. https://tayfunkayhan.wordpress.com/2019/07/26/chasing-triangles-in-a-tile-based-rasterizer/

[2]. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720

[3]. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[4]. https://github.com/zeux/meshoptimizer

[5]. https://llvm.org/docs/LangRef.html