# Ifrit-v2

GPU/CPU-Parallelized tile-based software rasterizer.

![img_demo0png](/img/img_demo0png.png)

![](img/img_demo1.png)

![](img/img_demo2.png)

![](img/img_demo3.png)



Successor to following repos:
 - [Ifrit](https://github.com/Aeroraven/Ifrit)
 - [Iris (TinyRenderer CPP)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris)
 - [Iris (TinyRenderer C#)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/TinyRenderer)



## Features

Overall framework for CUDA renderer pipeline (Some are different from its MT-CPU counterpart). Stages with asterisk mark are optional.

<img src="/img/overview.png" alt="overview" style="zoom: 67%;" />

**Note:** This project is NOT an exact replicate of hardware graphics pipeline (like TBDR architecture). Some behaviors are nondeterministic and some features incompatible under current implementation (like `Alpha Blending` which requires sorting primitives under parallel setting)

| Feature                                       | MT CPU Renderer | CUDA Renderer |
| --------------------------------------------- | --------------- | ------------- |
| Deterministic / Rendering Order               |                 |               |
| Performance / SIMD                            | √              |               |
| Performance / Overlapped Memory Transfer      |                 | √            |
| Performance / Dynamic Tile List               | √              | √ (2)        |
| Pipeline / Programmable Vertex Shader         | √              | √            |
| Pipeline / Programmable Fragment Shader       | √              | √            |
| Pipeline / Programmable Geometry Shader |  | √ (3) |
| Pipeline / Z Pre-Pass                         |                 | √            |
| Pipeline / Early-Z Test                       | √              | √            |
| Pipeline / Back Face Culling                  | √              | √            |
| Pipeline / Frustum Culling                    | √              | √            |
| Pipeline / Homogeneous Clipping               | √ (1)          | √ (1)        |
| Pipeline / Small Triangle Culling             |                 | √            |
| Pipeline / Perspective-correct Interpolation  | √              | √            |
| Texture / Basic Support                       |                 | √            |
| Presentation / Terminal ASCII                 | √              | √            |
| Presentation / Terminal Color                 | √              | √            |

(1) For performance consideration, only w-axis is considered 

(2) Causing latency issues

(3) Under testing, might be buggy



## Performance

Test performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored.

Note that some triangles might be culled or clipped in the pipeline. 

All tests were performed before git commit `7e6c34ad836842c02fcc9aa7dc89d5d01cd6cb66`. The result might not be the latest.

**Frame Rate**

| Model          | Triangles | CPU Single Thread* | CPU Multi Threads* | CUDA w/ Copy-back** | CUDA w/o Copy-back*** |
| -------------- | --------- | ------------------ | ------------------ | ------------------- | --------------------- |
| Yomiya         | 70275     | 38 FPS             | 80 FPS             | 123 FPS             | 2857 FPS              |
| Stanford Bunny | 208353    | 20 FPS             | 80 FPS             | 124 FPS             | 2272 FPS              |
| Khronos Sponza | 786801    | 2 FPS              | 10 FPS             | 125 FPS             | 500 FPS               |
| Intel Sponza   | 11241912  | 1 FPS              | 7 FPS              | 125 FPS             | 198 FPS               |

*. Under optimization 

**. Limited by PCIe performance

***. Might be influenced by other applications which utilize GPU



### Test Environment

- CPU: 12th Gen Intel(R) Core(TM) i9-12900H 
  - Test with 16 threads + AVX2 Instructions

- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (Kernel parameters are optimized for the SPECIFIC test environment)
- Shading: World-space normal



## Dependencies

- Hardware Requirements:
  - SSE
  - AVX2
  - CUDA 12.4
- Presentation Dependencies:
	- Terminal (Windows Terminal)
	- OpenGL 3.3
	- GLFW 3.3
	- GLAD
- Compile Dependencies:
	- <s>CMake 3.28</s>
	- MSVC (Visual Studio 2022)
		- C++17 is required
		- C++20 is recommended for best performance
	- NVCC



## Setup

Some dependencies should be prepared before compiling.

- Place `GLAD` dependency in `include\dependency\GLAD\glad\glad.h` and `include\dependency\GLAD\KHR\khrplatform.h`
- Place `sbt_image` in `include\dependency\sbt_image.h`



## Ongoing Plan

- Bug Fix & Testing
  - Resolution Change
- Tessellation
- Line Mode
- Texture LOD & Texture Sampler
- Multi-sampling
- Alpha Blending
- Mesh Shader
- Triangle Cluster & Cluster LOD



## References

For models / open source code references, check `licenses` folder.

[1]. https://tayfunkayhan.wordpress.com/2019/07/26/chasing-triangles-in-a-tile-based-rasterizer/

[2]. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720

[3]. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
