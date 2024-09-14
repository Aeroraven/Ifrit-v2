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
| 🚀 **Performance**                           |                                                              |                 |               |
| SIMD Instructions / SIMT                    | 🟥                                                            | ✅               | ✅             |
| Overlapped Memory Transfer                  | 🟥                                                            | 🟥               | ✅             |
| Dynamic Tile List                           | 🟥                                                            | ✅               | ✅             |
| 🔗 **Integration (Wrapper)**                 |                                                              |                 |               |
| C++ DLL                                     | 🟥                                                            | 🟦               | 🟥             |
| .NET Library (`C#`)                         | 🟥                                                            | 🟦               | 🟥             |
| 🔗 **Shader Language**                       |                                                              |                 |               |
| In-Application Class                        | ✅                                                            | ✅               | ✅             |
| SPIR-V Binary / HLSL                        | 🟥                                                            | 🟦 OrcJIT (2)    | 🟥             |
| SPIR-V Binary / GLSL                        | 🟥                                                            | 🟥               | 🟥             |
| 🔦 **Ray-tracer / Pipeline**                 |                                                              |                 |               |
| Acceleration Structure Traversal            | 🟥                                                            | 🟦               | 🟥             |
| Surface Area Heuristic                      | 🟥                                                            | ✅               | 🟥             |
| Programmable Ray Generation Shader          | 🟥                                                            | ✅               | 🟥             |
| Programmable Closest Hit Shader             | 🟥                                                            | ✅               | 🟥             |
| Programmable Miss Shader                    | 🟥                                                            | ✅               | 🟥             |
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
| Z Pre-Pass                                  | 🟥                                                            | 🟥               | ✅             |
| Early-Z Test                                | ✅                                                            | ✅               | ✅             |
| Late-Z Test (Depth Replacement & `discard`) | 🟥                                                            | 🟥               | ✅             |
| Scissor Test                                | 🟥                                                            | 🟥               | ✅             |
| Back Face Culling                           | ✅                                                            | ✅               | ✅             |
| Frustum Culling                             | 🟥                                                            | ✅               | ✅             |
| Homogeneous Clipping                        | 🟥                                                            | ✅               | ✅             |
| Small Triangle Culling                      | 🟥                                                            | 🟥               | ✅             |
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

Test performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored.

Note that some triangles might be culled or clipped in the pipeline. 

All tests were performed before git commit `7e6c34ad836842c02fcc9aa7dc89d5d01cd6cb66`. The result might not be the latest. Note that the introduction of `Shader Derivatives` degenerates the pipeline performance.

### **Frame Rate**

| Model          | Triangles | CPU Single Thread | CPU Multi Threads | CUDA w/ Copy-back* | CUDA w/o Copy-back** |
| -------------- | --------- | ----------------- | ----------------- | ------------------ | -------------------- |
| Yomiya         | 70275     | 38 FPS            | 80 FPS            | 123 FPS            | 2857 FPS             |
| Stanford Bunny | 208353    | 20 FPS            | 80 FPS            | 124 FPS            | 2272 FPS             |
| Khronos Sponza | 786801    | 2 FPS             | 10 FPS            | 125 FPS            | 500 FPS              |
| Intel Sponza   | 11241912  | 1 FPS             | 7 FPS             | 125 FPS            | 198 FPS              |

*. Limited by PCIe performance

**. Might be influenced by other applications which utilize GPU



### Test Environment

- CPU: 12th Gen Intel(R) Core(TM) i9-12900H 
  - Test with 16 threads + AVX2 Instructions

- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (Kernel parameters are optimized for the SPECIFIC test environment)
- Shading: World-space normal



## Dependencies

See [Requirements & Build Instructions ](./docs/requirement.md)for more details.



## Setup / Run

> Unexpected results observed in MinGW-compiled application

See [Requirements & Build Instructions ](./docs/requirement.md)for more details.



## Abstractions / Usage

See  [Usage](./docs/docs.md) for more details.



## Ongoing Plan

### Emergent Problem

- [ ] MinGW: Unexpected ray tracer result



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
  - [ ] Latency: JIT slows down execution
- [ ] Standardization
  - [ ] C++: `-Wignored-attributes` warnings in SIMD class  





## References

For models / open source code references, check `licenses` folder.

[1]. https://tayfunkayhan.wordpress.com/2019/07/26/chasing-triangles-in-a-tile-based-rasterizer/

[2]. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720

[3]. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[4]. https://github.com/zeux/meshoptimizer

[5]. https://llvm.org/docs/LangRef.html