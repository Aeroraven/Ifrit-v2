# Ifrit-v2

GPU/CPU-Parallelized tile-based software rasterizer.

![](img/img_demo1.png)

![](img/img_demo2.png)

![](img/img_demo3.png)



Successor to following repos:
 - [Ifrit](https://github.com/Aeroraven/Ifrit)
 - [Iris (TinyRenderer CPP)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris)
 - [Iris (TinyRenderer C#)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/TinyRenderer)



## Features

**Note:** This project is NOT a exact replicate of hardware graphics pipeline (like TBDR architecture). Some behaviors are nondeterministic and incompatible under current implementation (like `Alpha Blending` which requires sorting primitives under parallel setting)

| Feature                                       | MT CPU Renderer | CUDA Renderer |
| --------------------------------------------- | --------------- | ------------- |
| Deterministic / Rendering Order               | ×               | ×             |
| Performance / SIMD                            | √               |               |
| Performance / Overlapped Memory Transfer      |                 | √             |
| Performance / Dynamic Tile List               | √               | √ (2)         |
| Rendering / Programmable Vertex Shader        | √               | √             |
| Rendering / Programmable Fragment Shader      | √               | √             |
| Rendering / Z Pre-Pass                        | ×               | √             |
| Rendering / Back Face Culling                 | √               | √             |
| Rendering / Frustum Culling                   | √               | √             |
| Rendering / Homogeneous Clipping              | √ (1)           | √ (1)         |
| Rendering / Small Triangle Culling            | ×               | √             |
| Rendering / Perspective-correct Interpolation | √               | √             |
| Presentation / Terminal ASCII                 | √               | √             |
| Presentation / Terminal Color                 | √               | √             |

(1) For performance consideration, only w-axis is considered 

(2) Causing latency issues



## Performance

Test performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored.

Note that some triangles might be culled or clipped in the pipeline.

**Frame Rate**

| Model          | Triangles | CPU Single Thread* | CPU Multi-thread* | CUDA w/ Copy-back | CUDA w/o Copy-back** |
| -------------- | --------- | ------------------ | ----------------- | ----------------- | -------------------- |
| Yomiya         | 70275     | 38 FPS             | 80 FPS            | 123 FPS           | 1000 FPS^            |
| Stanford Bunny | 208353    | 20 FPS             | 80 FPS            | 124 FPS           | 850 FPS^             |
| Khronos Sponza | 786801    | 2 FPS              | 10 FPS            | 125 FPS           | 400 FPS              |
| Intel Sponza   | 11241912  | 1 FPS              | 7 FPS             | 125 FPS           | 180 FPS              |

*. Under optimization 

**. Might be influenced by other applications which utilize GPU

^. Peak value

### Test Environment

- CPU: 12th Gen Intel(R) Core(TM) i9-12900H 
  - Test with 16 threads + AVX2 Instructions

- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
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



## Ongoing Plan

- Texture



## References

