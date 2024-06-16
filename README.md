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

Test performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored.

Note that some triangles might be culled or clipped in the pipeline.

**Frame Rate**

| Model          | Triangles | CPU Single Thread* | CPU Multi-thread* | CUDA w/ Copy-back | CUDA w/o Copy-back** |
| -------------- | --------- | ------------------ | ----------------- | ----------------- | -------------------- |
| Yomiya         | 70275     | 38 FPS             | 80 FPS            | 123 FPS           | 1000 FPS^            |
| Stanford Bunny | 208353    | 20 FPS             | 80 FPS            | 124 FPS           | 850 FPS^             |
| Khronos Sponza | 786801    | 2 FPS              | 10 FPS            | 125 FPS           | 300 FPS              |
| Intel Sponza   | 11241912  | 1 FPS              | 7 FPS             | 125 FPS           | 155 FPS              |

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

- CPU Pipeline Optimization
  - Performance: [Pixel Proc] SIMD for tile-level pixel shading
  - Performance: [Pixel Proc] Z Pre-Pass
  - Performance: [General] Reduce fp division
- CUDA  Pipeline Optimization 
  - Performance: [Rasterizer 2] Memory Store Excessive
  - Performance: [Rasterizer 2] Not Selected Stall
  - Performance: [Rasterizer 1] Severe Latency Issue / Memory Bound
  - Bug: Coalesced Index Buffer
- Blending
- Scanline Rasterizer (IMR)

## References

