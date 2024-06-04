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

- **Performance**:
	- Multithreaded Rasterization
	- SIMD Vectorization
	- CUDA Acceleration (Incomplete)
		- Double Buffering / Overlapped Memory Transfer
		- Device Vector / Dynamic Array

- **Rendering**:
	- Homogeneous Space Clipping
	- Programmable VS/FS
	- Z Pre-Pass (CUDA-Only)

- **Presentation**:
	- Terminal Rendering (ASCII Characters/Color Image)



## Performance

Test performed on 2048x2048 RGBA FP32 Image + 2048x2048 F32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored.

| Model                     | CPU Single Thread | CPU Multi-thread | CUDA w/ Copy-back | CUDA w/o Copy-back |
| ------------------------- | ----------------- | ---------------- | ----------------- | ------------------ |
| Yomiya (70275 Triangles)  | 38 FPS            | 80 FPS           | 123 FPS           | 900 FPS            |
| Bunny (208353 Triangles)  | 20 FPS            | 80 FPS           | 124 FPS           | 625 FPS            |
| Sponza (786801 Triangles) | *                 | *                | 123 FPS           | 178 FPS            |

*. Geometry clipping stage in CPU renderer is buggy currently.



### Test Environment

- CPU: 12th Gen Intel(R) Core(TM) i9-12900H (Test with 16 threads)
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU



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

- Bug Fixing
	- Incorrect Homogeneous Space Clipping (CPU Part)
	- Incorrect Z Interpolation (CPU Part)
	- Incorrect Culling Order (CPU Part)
- CUDA Integration 
	- Performance: Pixel Processing Bottleneck
	- Fixed Subtile Size



## References

