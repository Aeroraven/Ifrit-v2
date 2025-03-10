# Ifrit-v2

> **注意事项 / Note**
> 
> 这还不是一个完整的版本，许多Commit中的内容并不是完整或正确的实现，已知的问题参见 TODO.md
> This is not a complete project, many features are not implemented correctly. To view known bugs, refer to TODO.md

Some toys about real-time rendering. Currently, it contains:

- **Soft-Renderer**: CUDA / Multithreaded CPU Software Rasterizer & Ray-tracer, with JIT support.
- **Syaro**: Deferred Renderer with Nanite-styled Cluster Level of Details.



| <center>Software Renderer / Mesh Shading</center>                        | <center>Software Renderer / CUDA Renderer</center> |
| ------------------------------------------------------- | --------------------------------- |
| ![](docs/img/img_demo3.png)                             | ![](docs/img/img_demo1.png)       |
| <center>**Syaro / Cull Rasterize Visibility Buffer (R32_UINT)**</center> | <center>**Syaro / Final Output**</center>          |
| ![](docs/img/syaro_clodvisb.png)                        | ![](docs/img/syaro_clod1.png)     |



Check  **[`gallery.md`](./docs/gallery.md)** for **[More Example Pictures](./docs/gallery.md)**

Successor to following repos: [Ifrit](https://github.com/Aeroraven/Ifrit), [Aria](https://github.com/Aeroraven/Aria) , [Iris (Tiny Renderer CPP)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris)  & [Iris (Tiny Renderer C#)](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/TinyRenderer)



## 1. Features Supported

### 1.1 Parallelized Soft Renderer

- Parallelized rasterization & ray-tracing pipeline, with GPU (CUDA) & Multithreaded CPU (SIMD) support
- Support mesh shading pipeline (mesh  shaders), and raytracing shaders (like miss shader)
- Support just-in-time compilation of HLSL SPIR-V shader code.
- Covers culling (including contribution culling), MSAA (8x), mipmapping,  anisotropic filtering and shader derivatives (`ddx` & `ddy`)
- Support texture sampling & cube mapping and texture lods.
- For implementation details and performance, check [here](./projects/softgraphics/readme.md)



### 1.2 Syaro: Virtual-Geometry-based Deferred Renderer

- Refactored version for [my original renderer](https://github.com/Aeroraven/Aria), improving pass management, synchronization primitives and descriptor bindings.

  - Bindless Descriptors
  - Dynamic Rendering

- Reproduced some features mentioned in Nanite's report: Two-pass occlusion culling, Mesh LoDs, Compute-shader-based SW rasterization.

- Some extra features supported:

  - Horizon-Based Ambient Occlusion
  - Cascaded Shadow Mapping
  - Temporal Anti-aliasings
  - Convolution Bloom (Fast Fourier Transform)

    

## 2. Setup / Run

> **WARN**: **Compiling files inside `dev` branch might yield UNEXPECTED result. (Known bugs are NOT resolved yet)**  Only x86-64 architecture `Windows` is tested. It mainly covers `Windows` and some `Linux` systems. Other operating systems like `MacOS` are NOT supported.

### 2.1 Clone the Repository

```bash
git clone https://github.com/Aeroraven/Ifrit-v2.git --recursive 
```



### 2.2 Install Dependencies

Following dependencies should be manually configured. Other dependencies will be configured via submodule.

- OpenGL >= 4.6 
- CMake >= 3.24

**Syaro**

- Vulkan SDK 1.3 (with shaderc combined)
  - Core Features 1.3
    - Or following extensions: `KHR_timeline_semaphore`, `KHR_dynamic_rendering`, `EXT_vertex_input_dynamic_state`, `EXT_color_write_enable`, `EXT_extended_dynamic_state3`,`EXT_extended_dynamic_state2`, `EXT_descriptor_indexing`, `KHR_spirv_1_4`, `EXT_host_query_reset`, `KHR_shader_float_controls`
  - with `EXT_mesh_shader` extension

**Soft Renderer** 

- LLVM 10 or LLVM 11 (Maybe higher version is OK, but LLVM 18 or higher might not work properly)
- CUDA >= 12.6 (If you have CUDA)



### 2.3 Quick Start 

> **Note:** 
>
> 1. CUDA support is temporarily not included in repo's CMake. 
> 2. Your compiler should support C++20 standards.
>
> **Under Refactoring, Linux GCC compilation MIGHT be unavailable now**

```shell
cmake -S . -B ./build
cmake --build ./build
```

To run the demo

- Download `lumberyard-bistro` , convert it into `gltf` format, then place it in the `project/demo/Asset/Bistro` directory, with dds textures in `textures` subfolder.

```shell
./bin/ifrit.demo
```





## 3. References & Acknowledgements

This project relies on following open-source projects. Corresponding licenses are in `licenses` folder.

| Repository                                                   | Usage                            | License              |
| ------------------------------------------------------------ | -------------------------------- | -------------------- |
| [stb](https://github.com/nothings/stb)                       | Image Parsing                    | MIT / Unlicensed     |
| [glfw3](https://github.com/glfw/glfw)                        | Window and Display support       | Zlib                 |
| [spirv-headers](https://github.com/KhronosGroup/SPIRV-Headers/) | SPIR-V Standard Reference        | MIT                  |
| [glad](https://github.com/Dav1dde/glad/)                     | OpenGL Header Generation.        | Generated files used |
| [llvm-project](https://github.com/llvm/llvm-project)         | JIT Runtime                      | Apache 2.0           |
| [meshoptimizer](https://github.com/zeux/meshoptimizer)       | Mesh Algorithm                   | MIT                  |
| [METIS](https://github.com/KarypisLab/METIS/)                | Mesh Algorithm / Graph Partition | Apache 2.0           |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | Memory Allocation                | MIT                  |
| [cereal](https://github.com/USCiLab/cereal)                  | Serialization                    | BSD-3                |
| [tinygltf](https://github.com/syoyo/tinygltf/tree/release)   | Model Loading                    | MIT                  |
| [stduuid](https://github.com/mariusbancila/stduuid)          | UUID                             | MIT                  |
| [spirv-reflect](https://github.com/KhronosGroup/SPIRV-Reflect) | Shader Reflection                | Apache 2.0           |
| [shaderc](https://github.com/google/shaderc?tab=License-1-ov-file#readme) | Shader Compilation               | Apache 2.0           |
| [sha1](https://github.com/vog/sha1)                          | Hash                             | Public domain        |
| [precomputed_atmospheric_scattering](https://github.com/ebruneton/precomputed_atmospheric_scattering) | Atmospheric Scattering           | BSD-3-Clause         |
| [gcem](https://github.com/kthohr/gcem)                       | Compile-time Math                | Apache 2.0           |
| [spdlog](https://github.com/gabime/spdlog.git)               | Logging                          | MIT                  |
| [fsr2](https://github.com/GPUOpen-Effects/FidelityFX-FSR2)   | Super Resolution                 | MIT                  |



Some tools are used during the development.

- [RenderDoc](https://renderdoc.org/), for debugging and inspecting.
- Nsight Graphics, for debugging and inspecting.
- Blender, for model conversion.



And some references that give inspirations. Thanks for their selfless dedications.

**Software Rasterization**：

1. https://tayfunkayhan.wordpress.com/2019/07/26/chasing-triangles-in-a-tile-based-rasterizer/
   1. https://github.com/NotCamelCase/Tyler
2. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720
3. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
4. https://llvm.org/docs/LangRef.html
5. https://www.mesa3d.org/
6. https://agner.org/optimize/

**Modern Graphics Pipeline**

1. https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf
2. https://qiutang98.github.io/post/%E5%AE%9E%E6%97%B6%E6%B8%B2%E6%9F%93%E5%BC%80%E5%8F%91/mynanite01_mesh_processor/
   1. https://github.com/qiutang98/chord/tree/master
3. https://jglrxavpok.github.io/2024/01/19/recreating-nanite-lod-generation.html
4. https://lesleylai.info/en/vk-khr-dynamic-rendering/
5. https://vulkan-tutorial.com/
6. https://poniesandlight.co.uk/reflect/island_rendergraph_1/
   1. https://github.com/tgfrerer/island
7. https://dev.to/gasim/implementing-bindless-design-in-vulkan-34no
8. https://github.com/KhronosGroup/Vulkan-Samples
9. https://www.elopezr.com/a-macro-view-of-nanite/
9. https://media.gdcvault.com/gdc2024/Slides/GDC+slide+presentations/Nanite+GPU+Driven+Materials.pdf
9. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720
9. https://developer.nvidia.com/orca/amazon-lumberyard-bistro

### 3.2 About naming
All names are chosen randomly from some characters.

- Ifrit: https://arknights.wiki.gg/wiki/Ifrit
- Syaro: https://gochiusa.fandom.com/wiki/Syaro_Kirima
  - Directly: https://osu.ppy.sh/beatmapsets/451250#osu/974142

## 4. License

It's licensed under [AGPL-v3 License (or later)](https://www.gnu.org/licenses/agpl-3.0.en.html), even it's hard to be integrated with server applications. The copy for license can be found in the root directory. 

