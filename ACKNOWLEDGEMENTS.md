
### 3.1 Dependencies

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
| [libktx](https://github.com/KhronosGroup/KTX-Software/blob/main/) * | Texture Compression              | Apache 2.0           |

*. License files can be obtained in submodule after you git clone this repository.



Some tools are used during the development.

- [RenderDoc](https://renderdoc.org/), for debugging and inspecting.
- Nsight Graphics, for debugging and inspecting.
- Blender, for model conversion.



And some references that give inspirations. Thanks for their selfless dedications.

**Overall**:
1. https://www.unrealengine.com/

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
10. https://media.gdcvault.com/gdc2024/Slides/GDC+slide+presentations/Nanite+GPU+Driven+Materials.pdf
11. https://www.slideshare.net/slideshow/optimizing-the-graphics-pipeline-with-compute-gdc-2016/59747720
12. https://developer.nvidia.com/orca/amazon-lumberyard-bistro
13. https://games-cn.org/games104-slides/

**Global Illumination**
1. https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf
2. https://zhuanlan.zhihu.com/p/696464007
3. https://zhuanlan.zhihu.com/p/522165652

**Coding**
1. https://github.com/TensorWorks/UE-Clang-Format
2. https://zhuanlan.zhihu.com/p/352723264


### 3.2 About naming
All names are chosen randomly from some characters.

- Ifrit: https://arknights.wiki.gg/wiki/Ifrit
- Syaro: https://gochiusa.fandom.com/wiki/Syaro_Kirima
  - Directly: https://osu.ppy.sh/beatmapsets/451250#osu/974142
