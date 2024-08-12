# Ifrit-v2

GPU/CPU-Parallelized tile-based software rasterizer.

![](./img/img_demo0png.png)

![](img/img_demo3.png)

![](img/img_demo1.png)

![](img/img_demo2.png)



Check  **[`gallery.md`](./gallery.md)** for **[More Example Pictures](./gallery.md)**



Successor to following repos:

 - [Ifrit](https://github.com/Aeroraven/Ifrit)
 - [Iris (TinyRenderer CPP)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris)
 - [Iris (TinyRenderer C#)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/TinyRenderer)





## Features

Overall framework for CUDA solid triangle renderer pipeline (Some are different from its MT-CPU counterpart). Stages with asterisk mark are optional. Tiling optimization is only applied for filled triangles.

<img src="./img/overview.png" alt="overview" style="zoom: 67%;" />

**Note:** This project is NOT an exact replicate of hardware graphics pipeline (like IMR or TBDR architecture). 

| Feature                                     | [Iris Renderer](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris) | MT CPU Renderer | CUDA Renderer |
| ------------------------------------------- | ------------------------------------------------------------ | --------------- | ------------- |
| **Basic**                                   |                                                              |                 |               |
| Rendering Order                             | √                                                            | √               | √ (3)         |
| **Performance**                             |                                                              |                 |               |
| SIMD Instructions / SIMT                    |                                                              | √               | √             |
| Overlapped Memory Transfer                  |                                                              |                 | √             |
| Dynamic Tile List                           |                                                              | √               | √             |
| **Pipeline**                                |                                                              |                 |               |
| Programmable Vertex Shader                  | √                                                            | √               | √             |
| Programmable Fragment Shader                | √                                                            | √               | √             |
| Programmable Geometry Shader                |                                                              |                 | ▲             |
| Programmable Mesh Shader                    |                                                              |                 | ▲             |
| Programmable Task Shader                    |                                                              |                 | ▲             |
| Alpha Blending                              |                                                              | √               | √             |
| Depth Testing                               | √                                                            | √               | √             |
| Depth Function                              |                                                              | √               | √             |
| Z Pre-Pass                                  |                                                              |                 | √             |
| Early-Z Test                                | √                                                            | √               | √             |
| Late-Z Test (Depth Replacement & `discard`) |                                                              |                 | √             |
| Scissor Test                                |                                                              |                 | √             |
| Back Face Culling                           | √                                                            | √               | √             |
| Frustum Culling                             |                                                              | √               | √             |
| Homogeneous Clipping                        |                                                              | √ (1)           | √ (1)         |
| Small Triangle Culling                      |                                                              |                 | √             |
| Perspective-correct Interpolation           |                                                              | √               | √             |
| Shader Derivatives `dFdx` `dFdy`            |                                                              |                 | ▲ (2)         |
| **Polygon Mode**                            |                                                              |                 |               |
| Filled Triangle                             | √                                                            | √               | √             |
| Line (Wireframe)                            |                                                              |                 | ▲             |
| Point                                       |                                                              |                 | ▲             |
| **Texture**                                 |                                                              |                 |               |
| Basic Support (Sampler)                     |                                                              |                 | √             |
| Blit                                        |                                                              |                 | √             |
| Mipmap                                      |                                                              |                 | √             |
| Filter                                      |                                                              |                 | √             |
| Sampler Address Mode                        |                                                              |                 | √             |
| LOD Bias                                    |                                                              |                 | √             |
| Anisotropic Filtering                       |                                                              |                 | ▲ (4)         |
| Cube Map                                    |                                                              |                 | √             |
| **Presentation**                            |                                                              |                 |               |
| Terminal ASCII                              |                                                              | √               | √             |
| Terminal Color                              |                                                              | √               | √             |

(1) For performance consideration, only w-axis is considered 

(2) Shader derivatives are now only available for the filled triangle polygon mode. Shader derivatives are calculated in `2x2` quads, so precision might matter.

(3) Only works when `Alpha Blending` is enabled.

(4) Only works when `texture` shader function is called.

▲ Limited support.



### Supported Feature Details

- Sampler Filter :`IF_FILTER_NEAREST`, `IF_FILTER_LINEAR`
- Sampler Address Mode: `IF_SAMPLER_ADDRESS_MODE_REPEAT`, `IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE`, `IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER` , `IF_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT`, `IF_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE`



## Performance

Test performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored.

Note that some triangles might be culled or clipped in the pipeline. 

All tests were performed before git commit `7e6c34ad836842c02fcc9aa7dc89d5d01cd6cb66`. The result might not be the latest. Note that the introduction of `Shader Derivatives` degenerates the pipeline performance.

**Frame Rate**

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

#### Minimal Requirement

- **Display Dependencies**: 
  - OpenGL (GLFW3.3 + GLAD)
- **Compilation Dependencies:** One of following environments. Requires `c++20` support.
  - MSVC 19.29 + Visual Studio 2022 
  - CMake 3.28 + GCC 13.2 (MinGW Included) `[CUDA Support is unknown]`

#### Recommended Requirement

- **Hardware Requirements:**  
  - CUDA 12.4 Supports
  - AVX2 Support
- **Display Dependencies**: 
  - OpenGL (GLFW3.3 + GLAD)
- **Compilation Dependencies:** Requires `c++20` support.
  - MSVC 19.29 + Visual Studio 2022 



## Setup / Run

#### Dependency Installation

Some dependencies should be prepared before compiling.

- Place `GLAD` dependency in `include\dependency\GLAD\glad\glad.h` and `include\dependency\GLAD\KHR\khrplatform.h`
- Place `sbt_image` in `include\dependency\sbt_image.h`

Change CUDA path and GLFW3 library path in `CMakeLists.txt` 



#### Compile using G++ / MinGW

Follow instructions to build

```cmake
cmake -S . -B ./build
cd build
make
```



#### Compile using Visual Studio 2022 (MSVC)

Directly open `Ifrit-v2x.sln` in Visual Studio 2022.

Edit the property sheet to help the linker find CUDA and GLFW3 library file. Then press `F5`





## Abstractions / Usage

See `DOCS.md` for more details.



## Ongoing Plan

- Tessellation
- <s>Line Mode</s>
- <s>Texture LOD & Texture Sampler</s>
  - <s>Shader Derivatives</s>
  - <s>Anisotropic Filtering</s>
  - <s>Dynamic LOD Selection & Texture Bias</s>
  - <s>Cubic Texture</s>
  - Tiling
- Multi-sampling
- <s>Alpha Blending</s>

  - <s>Sorting</s>
- <s>Mesh Shader</s>
- Input Topology
- Triangle Cluster & Cluster LOD
- Known Issues

  - <s>Issue: Faults after resolution change</s>
  - Overdraw: Point mode with index buffer
  - Latency: Excessive global atomics in line mode
  - Issue: Nondeterministic behaviors in wireframe/point mode 
  - <s>Issue: Artifacts in low resolution scenario </s>





## References

For models / open source code references, check `licenses` folder.

[1]. https://tayfunkayhan.wordpress.com/2019/07/26/chasing-triangles-in-a-tile-based-rasterizer/

[2]. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720

[3]. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[4]. https://github.com/zeux/meshoptimizer