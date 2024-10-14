# Ifrit-v2

GPU/CPU-Parallelized tile-based software rasterizer.

![](./docs/img/img_demo0png.png)

![](docs/img/img_demo3.png)

![](docs/img/img_demo1.png)

![](docs/img/img_demo2.png)



Check  **[`gallery.md`](./docs/gallery.md)** for **[More Example Pictures](./gallery.md)**



Successor to following repos:

 - [Ifrit](https://github.com/Aeroraven/Ifrit)
 - [Iris (TinyRenderer CPP)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris)
 - [Iris (TinyRenderer C#)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/TinyRenderer)





## Features

Overall framework for CUDA solid triangle renderer pipeline (Some are different from its MT-CPU counterpart). Stages with asterisk mark are optional. Tiling optimization is only applied for filled triangles.

<img src="./docs/img/overview.png" alt="overview" style="zoom: 67%;" />

**Note:** This project is NOT an exact replicate of hardware graphics pipeline (like IMR or TBDR architecture). It's just a toy or a tutorial for the basic understanding.

✅ Available | 🟦 Limited  Support (Under Testing) | 🟨 Severely Unstable (Under Testing) | 🟥 TODO

| Feature                                                 | [Iris Renderer](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris) | MT CPU Renderer | CUDA Renderer |
| ------------------------------------------------------- | ------------------------------------------------------------ | --------------- | ------------- |
|                                                         |                                                              |                 |               |
| 🔗 **Integration (Wrapper)**                             |                                                              |                 |               |
| C++ DLL                                                 | 🟥                                                            | 🟦               | 🟥             |
| .NET Library (`C#`)                                     | 🟥                                                            | 🟦               | 🟥             |
| 🔗 **Shader Language**                                   |                                                              |                 |               |
| In-Application Class                                    | ✅                                                            | ✅               | ✅             |
| SPIR-V Binary / HLSL                                    | 🟥                                                            | 🟦 OrcJIT (2)    | 🟥             |
| SPIR-V Binary / GLSL                                    | 🟥                                                            | 🟥               | 🟥             |
| 🚀 **Ray-tracer / Performance**                          |                                                              |                 |               |
| SIMD Instructions / SIMT                                | 🟥                                                            | ✅               | 🟥             |
| Acceleration Structure (BVH)                            | 🟥                                                            | ✅               | 🟥             |
| Lock-free Synchronization                               | 🟥                                                            | ✅               | ⬜             |
| 🔦 **Ray-tracer / Pipeline**                             |                                                              |                 |               |
| Acceleration Structure Traversal                        | 🟥                                                            | 🟦               | 🟥             |
| Surface Area Heuristic                                  | 🟥                                                            | ✅               | 🟥             |
| Programmable Ray Generation Shader                      | 🟥                                                            | ✅               | 🟥             |
| Programmable Closest Hit Shader                         | 🟥                                                            | ✅               | 🟥             |
| Programmable Miss Shader                                | 🟥                                                            | ✅               | 🟥             |
| 🚀 **Rasterization / Performance**                       |                                                              |                 |               |
| SIMD Instructions / SIMT                                | 🟥                                                            | ✅               | ✅             |
| Overlapped Memory Transfer                              | ⬜                                                            | ⬜               | ✅             |
| Dynamic Tile List                                       | 🟥                                                            | ✅               | ✅             |
| Lock-free Synchronization                               | 🟥                                                            | ✅               | ⬜             |
| 💡 **Rasterization / Basic**                             |                                                              |                 |               |
| Rendering Order                                         | ✅                                                            | ✅               | ✅             |
| 💡 **Rasterization / Pipeline**                          |                                                              |                 |               |
| Programmable Vertex Shader                              | ✅                                                            | ✅               | ✅             |
| Programmable Pixel Shader                               | ✅                                                            | ✅               | ✅             |
| Programmable Geometry Shader                            | 🟥                                                            | 🟥               | 🟦             |
| Programmable Mesh Shader                                | 🟥                                                            | 🟥               | 🟦             |
| Programmable Task Shader                                | 🟥                                                            | 🟥               | 🟦             |
| Alpha Blending                                          | 🟥                                                            | ✅               | ✅             |
| Depth Testing                                           | ✅                                                            | ✅               | ✅             |
| Depth Function                                          | 🟥                                                            | ✅               | ✅             |
| Z Pre-Pass / Tag Buffer                                 | 🟥                                                            | ✅               | ✅             |
| Early-Z Test                                            | ✅                                                            | ✅               | ✅             |
| Late-Z Test (Depth Replacement & `discard`)             | 🟥                                                            | 🟥               | ✅             |
| Scissor Test                                            | 🟥                                                            | 🟥               | ✅             |
| Back Face Culling                                       | ✅                                                            | ✅               | ✅             |
| Frustum Culling                                         | 🟥                                                            | ✅               | ✅             |
| Homogeneous Clipping                                    | 🟥                                                            | ✅               | ✅             |
| Small Triangle Culling                                  | 🟥                                                            | ✅               | ✅             |
| Perspective-correct Interpolation                       | 🟥                                                            | ✅               | ✅             |
| Shader Derivatives `dFdx` `dFdy`<br/>Helper Invocations | 🟥                                                            | 🟨               | 🟦             |
| Multi-sampling                                          | 🟥                                                            | 🟥               | 🟦 8x MSAA     |
| 💡 **Rasterization / Polygon Mode**                      |                                                              |                 |               |
| Filled Triangle                                         | ✅                                                            | ✅               | ✅             |
| Line (Wireframe)                                        | 🟥                                                            | 🟥               | 🟦             |
| Point                                                   | 🟥                                                            | 🟥               | 🟦             |
| 🖼️ **Texture**                                           |                                                              |                 |               |
| Basic Support (Sampler)                                 | 🟥                                                            | 🟥               | ✅             |
| Blit                                                    | 🟥                                                            | 🟥               | ✅             |
| Mipmap                                                  | 🟥                                                            | 🟥               | ✅             |
| Filter                                                  | 🟥                                                            | 🟥               | ✅             |
| Sampler Address Mode                                    | 🟥                                                            | 🟥               | ✅             |
| LOD Bias                                                | 🟥                                                            | 🟥               | ✅             |
| Anisotropic Filtering                                   | 🟥                                                            | 🟥               | 🟦             |
| Cube Map                                                | 🟥                                                            | 🟥               | ✅             |
| 🖥️ **Presentation**                                      |                                                              |                 |               |
| Terminal ASCII                                          | 🟥                                                            | ✅               | ✅             |
| Terminal Color                                          | 🟥                                                            | ✅               | ✅             |

(1) Shader derivatives are now only available for the filled triangle polygon mode. Shader derivatives are calculated in `2x2` quads, so precision might matter.

(2) Partial instructions are supported. Only available for binaries produced by `glslc` or `dxc`

### Supported Feature Details

- Sampler Filter :`IF_FILTER_NEAREST`, `IF_FILTER_LINEAR`
- Sampler Address Mode: `IF_SAMPLER_ADDRESS_MODE_REPEAT`, `IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE`, `IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER` , `IF_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT`, `IF_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE`



## Performance

### Frame Rate Comparison (FPS)  Version 2



#### Performance Comparison For CPU Software Rendering

Tests performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored. Note that some triangles **might be culled or clipped** in the pipeline. 

| Model                             | Yomiya | Bunny | Khronos Sponza | Intel Sponza |
| --------------------------------- | ------ | ----- | -------------- | ------------ |
| **Windows (MSVC)**                |        |       |                |              |
| Ifrit-v2 MTCPU                    | 153    | 126   | 50             | 24           |
| **WSL2 (Ubuntu 20.04 LTS / GCC)** |        |       |                |              |
| Ifrit-v2 MTCPU                    | 128    | 102   | 43             | 20           |
| Mesa3D softpipe                   | 13     | 3     | *              | *            |
| Mesa3D llvmpipe                   | 152    | 76    | 50             | 8            |

 *. Frame time larger than 1000ms (FPS<1)



#### Influence of Attachment Size

Tests performed on multi-thread CPU renderer (1 master + 16 workers), with just-in-time(JIT) compilation of Vulkan-specific HLSL shaders (compiled in SPIR-V binary format). All attachments are in `linear` tiling mode and `float32` mode. 

| Model                                     | 512 x 512 | 1024 x 1024 | 2048 x 2048 | 4096 x 4096 |
| ----------------------------------------- | --------- | ----------- | ----------- | ----------- |
| Kirara / Genshin Impact (37 k)            | 1219      | 480         | 124         | 28          |
| Evil Neurosama (55.9 k)                   | 606       | 398         | 120         | 31          |
| Flandre Scarlet / Touhou Project (96.1 k) | 502       | 237         | 82          | 19          |
| Miyako / Blue Archive (346.1 k)           | 106       | 72          | 43          | 13          |



#### Influence of Multithreading, Optimization and Execution Strategy

Tests performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored. Note that some triangles **might be culled or clipped** in the pipeline. 

| Model                                         | Yomiya     | Bunny      | Khronos Sponza | Intel Sponza |
| --------------------------------------------- | ---------- | ---------- | -------------- | ------------ |
| **Architectural Optimization & JIT**          |            |            |                |              |
| Triangles                                     | 70275      | 208353     | 786801         | 11241912     |
| Single Thread CPU Baseline v1                 | 38         | 20         | 2              | 1            |
| Multi Thread CPU Baseline v1                  | 80         | 80         | 10             | 2            |
| CUDA Baseline v1                              | 2857       | 2272       | 500            | 198          |
| ST CPU Optimized v2 (C++ / SPIR-V HLSL)       | 56 (+47%)  | 37 (+85%)  | 7 (+250%)      | 4 (+300%)    |
| MT CPU Optimized v2 (C++ / SPIR-V HLSL)       | 153 (+91%) | 125 (+56%) | 50 (+400%)     | 24 (+1100%)  |
| ST CPU Optimized v2 (C++ / Class Inheritance) | 56 (+47%)  | 37 (+85%)  | 7 (+250%)      | 4 (+300%)    |
| MT CPU Optimized v2 (C++ / Class Inheritance) | 153 (+91%) | 126 (+58%) | 50 (+400%)     | 24 (+1100%)  |
| **Pipeline Optimization**                     |            |            |                |              |
| Optimized                                     | 153        | 126        | 50             | 24           |
| w/o Tag Buffering                             | 119 (-22%) | 98 (-22%)  | 24 (-52%)      | 17 (-29%)    |
| w/o Hierarchical Tiling                       | 111 (-26%) | 86 (-32%)  | 19 (-62%)      | 17 (-29%)    |

※ **C++ Class**: shaders are coded and compiled ahead-of-time, using virtual inheritance.

※ **SPIR-V HLSL (C++)**: all shader codes are compiled into binary form using `glslc`. HLSL source codes are written in `Vulkan-specific` style. Just-in-time (JIT) compilation uses LLVM 10 as backend and manual IR mapping (Shared library is compiled with `mingw-w64`). App runs in `msvc`.



#### Influence of Data Parallelism

Tests performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored. Note that some triangles **might be culled or clipped** in the pipeline. SPIR-V HLSL is used for shading.

| Model          | ST   | MT   | MT + SSE (128) | MT + AVX2 (256) | Mem Bound | CUDA | PCIe Bound |
| -------------- | ---- | ---- | -------------- | --------------- | --------- | ---- | ---------- |
| Khronos Sponza | 4    | 33   | 43             | 50              | 128       | 500  | 125        |

※ **Single Thread (ST) and Multi Threads (MT)**: 16 workers (+1 master) are used for multi thread setting

※ **Data Parallelism**: 

- **SSE/AVX**: FP32 numbers are organized into 4-element vectors with SSE instruction supports and 8-element vectors with AVX2 instruction support. Fused multiply add instructions (`FMA`) are used in both setting. 

※ **Performance Bound**: 

- **Mem Bound**: The maximal fps measured when the attachments are painted using memory set functions (`rep stosb` instruction). This measures the upper performance limit of a CPU renderer.
- **PCIe Bound**: The maximal fps measured when the attachment are copied back from local device (CUDA device) to host device (CPU). This measures the upper performance limit of a CUDA renderer at the final step (without graphics API interop considerations)





### **Frame Rate Comparison (FPS)  Version 1**

See [Performance History](./docs/performance.md)



### Test Environment

- CPU: 12th Gen Intel(R) Core(TM) i9-12900H 
  - Test with 17 threads (1 master + 16 workers) + AVX2 Instructions
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (Kernel parameters are optimized for the SPECIFIC test environment)
- Shading: World-space normal
- Compiler:  MSVC 19.29 + NVCC 12.6 (with O2 optimization)



## Dependencies

See [Requirements & Build Instructions ](./docs/requirement.md)for more details.



## Setup / Run

### Quick Start (GCC / MinGW-w64)

> Warning: Please ensure that dependencies are installed. **And this will ignore all CUDA codes in the project.** 
>
> NOTE: Some paths / packages should be configured manually before cmake

```shell
# G++ (Linux)
cmake -S . -B ./build
cd build
make
# MinGW-w64 (Windows)
cmake -S . -B ./build -G "MinGW Makefiles"
cd build
mingw32-make
```

To run the demo

```shell
# For linux
export LD_LIBRARY_PATH=/path/to/Ifrit.Components.LLVMExec.so;$LD_LIBRARY_PATH
./core/bin/IfritMain
```



### Complete Build Options 

See [Requirements & Build Instructions ](./docs/requirement.md)for more details.



## Abstractions / Usage

See  [Usage](./docs/docs.md) for more details.



## TODO

### Short-term Plan

- [ ] MTCPU: Basic Texture Support
- [ ] MTCPU: Branch Divergence Handling Assurance in Quads

### Long-term Plan

- [ ] Tessellation

- [ ] Texture Tiling

- [ ] Multi-sampling: Blending Integration

- [ ] Input Topology

- [ ] Triangle Cluster & Cluster LOD

- [ ] Known Issues
  
  - [ ] Overdraw: Point mode with index buffer
  - [ ] Latency: Excessive global atomics in line mode
  - [ ] Latency: Memory access pattern in MSAA
  - [ ] Latency: JIT slows down execution (in raytracer)
  
  





## References

For models / open source code references, check `licenses` folder. Thanks for following resources.

[1]. https://tayfunkayhan.wordpress.com/2019/07/26/chasing-triangles-in-a-tile-based-rasterizer/

[2]. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720

[3]. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[4]. https://github.com/zeux/meshoptimizer

[5]. https://llvm.org/docs/LangRef.html

[6]. https://www.mesa3d.org/