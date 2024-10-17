# Ifrit-v2

GPU/CPU-Parallelized tile-based software rasterizer.

| ![](docs/img/img_demo3.png) | ![](docs/img/img_demo1.png) |
| --------------------------- | --------------------------- |

Check  **[`gallery.md`](./docs/gallery.md)** for **[More Example Pictures](./docs/gallery.md)**

Successor to following repos: [Ifrit](https://github.com/Aeroraven/Ifrit), [Aria](https://github.com/Aeroraven/Aria) , [Iris (Tiny Renderer CPP)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris)  & [Iris (Tiny Renderer C#)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/TinyRenderer)



## 1. Structure

The project is organized into following parts.

- **`softrenderer`**: CUDA and CPU multithreaded SIMD software rasterizer & ray-tracer.
  - Covers culling, mesh shading pipeline (mesh shader), MSAA (up to 8x), mipmap, anisotropic filtering, SPIR-V JIT execution and shader derivatives.
  - CPU renderer is organized in TBDR-like manner, utilize AVX2 instructions and tag buffer (with early-z) for performance gain.
  - For implementation details and performance, check [here](./projects/softrenderer/readme.md)
- **`vkrenderer`**: Vulkan renderer, intended to be the refactored version for [Aria](https://github.com/Aeroraven/Aria).
- **`meshproclib`**: Mesh algorithms.
- **`ircompile`**: LLVM JIT compilation for shader codes.
- **`display`**:  Presentation and window surface supporting utilities.



## 2. Setup / Run

> WARN: **Compiling files inside `dev` branch might yield unexpected result.**  Only x86-64 architecture `Windows` and  `Ubuntu` are tested. It mainly covers `Windows` and some `Linux` systems. Other operation systems like `MacOS` are NOT supported.

### 2.1 Clone the Repository

```bash
git clone https://github.com/Aeroraven/Ifrit-v2.git --recursive  
```



### 2.2 Quick Start (GCC / MinGW-w64)

> Note: CUDA support is temporarily not included in repo's CMake. Your compile should support C++20 standard.
>
> Please ensure that `find_package` can find  `vulkan`, `llvm>=10,<12` and `glfw3==3.3`. Otherwise, please manually change the fallback path in `CMakeLists.txt`. To install `llvm` and `glfw3`, check Complete Build Options. 
>
> **Under Refactoring, Linux GCC compilation MIGHT be  unavailable now**

```shell
cmake -S . -B ./build
cmake --build ./builds
```

To run the demo

```shell
./bin/ifrit.demo
```



### 2.3 Quick Start (MSVC)

> Note: Path to `CUDA`, `llvm` and `glfw3`should be manually configured in the property sheets. To install `cuda`, `llvm` and `glfw3`, check Complete Build Options. **Under Refactoring, MSVC compilation is not available now**

Just open `Ifrit-v2x.sln` in `Visual Studio` and compile the project.



### 2.4  Complete Build Options 

See [Requirements & Build Instructions ](./docs/requirement.md)for more details.



## 3. References & Acknowledgements

This project relies on following open-source projects. Corresponding licenses are in `licenses` folder.

- [stb](https://github.com/nothings/stb), for image parsing.

- [glfw3](https://github.com/glfw/glfw), for window and display support.

- [spirv-headers](https://github.com/KhronosGroup/SPIRV-Headers/), for spirv standard reference.

- [glad](https://github.com/Dav1dde/glad/), for opengl header generation.

- [llvm-project](https://github.com/llvm/llvm-project), for just-in-time compilation support

- [meshoptimizer](https://github.com/zeux/meshoptimizer), for mesh operations

- [METIS](https://github.com/KarypisLab/METIS/), for graph partitioning

  

And some references that give inspirations:

1. https://tayfunkayhan.wordpress.com/2019/07/26/chasing-triangles-in-a-tile-based-rasterizer/
   1. https://github.com/NotCamelCase/Tyler
2. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720
3.  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
4. https://llvm.org/docs/LangRef.html
5. https://www.mesa3d.org/
6. https://agner.org/optimize/
7. https://qiutang98.github.io/post/%E5%AE%9E%E6%97%B6%E6%B8%B2%E6%9F%93%E5%BC%80%E5%8F%91/mynanite01_mesh_processor/
   1. https://github.com/qiutang98/chord/tree/master
8. https://jglrxavpok.github.io/2024/01/19/recreating-nanite-lod-generation.html





