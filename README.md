# Ifrit-v2

GPU/CPU-Parallelized tile-based software rasterizer.

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


## Dependencies

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
	
- Hardware Requirements:
	- SSE
	- AVX2
	- CUDA 12.4

## Ongoing Plan
- CUDA Integration 
	- Performance: LSB Stall in Primitive Assembly Stage
	- Performance: LSB Stall in Pixel Shading Stage
	- Performance: Barrier Stall in Rasterization Stage
	- Performance: LG in Homogeneous Clipping Stage