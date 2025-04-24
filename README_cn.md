# Ifrit-v2

[English](./README.md) | **简体中文**


一些关于Real-Time Rendering的小玩具，目前包括内容：

- **Soft-Renderer**: CUDA / 多线程SIMD 软件光栅和软件光线追踪, 包含JIT支持
- **Experimental Renderer**:
  - **Syaro**: 基于Nanite风格的虚拟几何体的延迟着色渲染器 （正在开发）
  - **Ayanami**: 基于Probe和软件光线追踪(SDF)的全局光照. (正在开发)




| Soft Renderer / Mesh 着色                                   | Soft Renderer / CUDA Renderer                             |
| ----------------------------------------------------------- | --------------------------------------------------------- |
| ![](docs/img/img_demo3.png)                                 | ![](docs/img/img_demo1.png)                               |
| **Syaro / Cull Rasterize 可见性缓冲 (R32_UINT)**            | **Syaro / 最终输出**                                      |
| ![](docs/img/syaro_clodvisb.png)                            | ![](docs/img/syaro_clod1.png)                             |
| **Syaro / 计时**                                            | **Soft Renderer / 着色器导数**                            |
| <img src="docs/img/img_syaroperf.jpg" style="zoom: 67%;" /> | ![](docs/img/soft_dx1.png)                                |
| **Ayanami / GDF Object Grids + Surface Cache Lookup***      | **Ayanami / 全局距离场 (GDF)\***                          |
| ![](docs/img/ayanami_objgrid_exp1.png)                      | ![](docs/img/ayanami_globaldf2.png)                       |
| **Soft Renderer / Profile** (Nsight Compute)                | **Ayanami / Debug (追踪层级结构，未完成) \*** (RenderDoc) |
| <img src="docs/img/soft_nscp.png" style="zoom:80%;" />      | <img src="docs/img/aya_diag.png" style="zoom:80%;" />     |

**访问[`GALLERY.md`](./GALLERY.md) 查看更多图片.**

要可视化 Syaro's LoD 的变化过程, 可以参考 `docs/img/syaro_lod.mkv`

<br/><br/>

Ifrit-v2的构建基于下方的个人仓库: 

- [Aria](https://github.com/Aeroraven/Aria): 一些关于 WebGL2 和 Vulkan 的玩具
  - 其中一些[特性](#4-future-plans)将在后续考虑整合 

- [Ifrit-v1](https://github.com/Aeroraven/Ifrit): 一个简易的控制台绘图支持的实现尝试（针对于TJ的Java的课程设计）
  - Ifrit-v2 同样支持软光栅结果在控制台的输出，可见上方图库链接
- Iris ([C#](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/TinyRenderer) / [C++](https://github.com/Aeroraven/Stargazer/tree/main/ComputerGraphics/Iris)): 个人版 Tiny Renderer 复刻.



## 1. 特性 / Features

### 1.1 Parallelized Soft Renderer

- 并行光栅化与射线追踪管线，支持GPU (CUDA)和多线程CPU (SIMD)
- 支持网格着色管线（mesh shaders）以及射线追踪着色器（例如miss shader）
- 支持HLSL SPIR-V着色器代码的即时编译
- 涵盖剔除（包括Contribution Culling）、MSAA（8倍）、mipmapping、各向异性过滤和着色器微分（`ddx`和`ddy`）
- 支持纹理采样、立方体映射和纹理LOD
- 完整的性能报告和功能，参见[这里](./projects/softgraphics/readme.md)



### 1.2 Experimental Renderer

对 [之前个人的小玩具](https://github.com/Aeroraven/Aria) 的重构版本, 优化了对Pass/Descriptor的管理.

- Bindless Descriptors
- Dynamic Rendering
- Render Hardware Interface
- Render Dependency Graph
  - 无锁资源池和初步的资源复用
  - 简单的RDG资源生命周期管理和状态追踪
- Task System
- 初步的纹理压缩支持

#### 1.2.1 Syaro: Virtual-Geometry-based Deferred Renderer

- 复现了Nanite报告中的部分特征，包括：两遍剔除，Mesh细节层级，动态LoD选择，Compute Shader软渲染，简单的材质系统
- 额外支持的功能:
  - HBAO环境光遮蔽
  - 级联阴影
  - 时序抗锯齿
  - 卷积泛光

#### 1.2.2 Ayanami: Maybe Something about Global Illumination

- 部分实现细节，参阅 [这里](./include/ifrit.shader/Ayanami/Readme.md)

- 完全由RDG驱动

- 当前已经初步实现的内容：
  - 距离场生成和距离场相关内容*
    - 距离场阴影剔除
    - 距离场软阴影 DFSS
    - BC4压缩
    
  - Surface Cache*
    - Object Grids 
    
  - 光照探针*
    - 自适应Screen Probe放置
    
    - Screen Probe追踪
      - 屏幕空间光线追踪 (SSGI+HiZ)
      - Mesh和全局距离场追踪 (Grid Cull+Ray Marching)
      
      

*. 由于还在功能实现阶段，性能优化还暂时未覆盖到


## 2. Setup / Run

### 2.1  拉取本仓库

```bash
git clone https://github.com/Aeroraven/Ifrit-v2.git --recursive 
```

### 2.2 安装依赖

下面的依赖需要手动配置，其他的依赖通过Git子模块安装（确保拉取使用了`--recursive`）

- OpenGL >= 4.6 
- CMake >= 3.24
- MSVC >= 19.29 (`cpp20` Support Required)

**Ifrit Runtime (Syaro/Ayanami)**

- Vulkan SDK 1.3 (with shaderc combined)
  - Core Features 1.3
  - with `EXT_mesh_shader` extension
  - with `EXT_shader_image_atomic_int64` extension

**Ifrit Soft Renderer** 

- LLVM >= 11.0
- CUDA >= 12.5 (Optional)
  - 已知编译器兼容性问题： CUDA 12.4 和 MSVC (问题在CUDA 12.5修复)

### 2.3  编译和运行

> 软光栅的Demo，请切换到其他分支或历史Commit

```shell
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build ./build
```



运行Demo

- 下载 `lumberyard-bistro`，将其转换为 `gltf` 格式（命名为 `untitled.gltf`），然后放置在 `project/demo/Asset/Bistro` 目录中，并将 dds 纹理存放在 `textures` 子文件夹中。

```shell
./bin/ifrit.demo.syaro.exe
```



## 3. 结构

源文件可以拆分成下列模块

| 模块               | 功能说明                                                     |
| ------------------ | ------------------------------------------------------------ |
| ifrit.core         | 基本定义、日志记录、序列化、类型工具（如编译时工具）         |
| ifrit.core.math    | 用于SIMD和性能优化intrinsics的辅助函数<br/>基本线性代数支持  |
| ifrit.runtime      | 渲染器实现<br/>提供网格、资源、组件和渲染的基本支持          |
| ifrit.demo         | 演示                                                         |
| ifrit.display      | 特定平台的窗口支持<br/>为渲染器提供视图层，如软渲染器的控制台显示 |
| ifrit.external     | 外部依赖构建<br/>包含FSR2                                    |
| ifrit.ircompile    | JIT运行时后端<br/>基于LLVM                                   |
| ifrit.imaging      | 图像处理工具<br/>包括一些纹理压缩工具                        |
| ifrit.meshproc     | 网格处理算法和CPU加速结构<br/>包括网格聚类剔除数据生成、网格自动LOD及网格级别有符号距离场生成 |
| ifrit.rhi          | 与后端无关的渲染硬件接口                                     |
| ifrit.softgraphics | 软渲染器实现，同时提供多线程CPU和CUDA版本                    |
| ifrit.vkgraphics   | Vulkan后端                                                   |



## 4. References & Acknowledgements

请参阅 ACKNOWLEDGEMENTS.md 以获取更多细节。

某些思路可能借鉴自 Unreal Engine（或其相关的 SIG 或 GDC 演讲）。然而，由于许可兼容性问题（AGPL-v3 与 Unreal Engine 的 EULA），代码并未直接从 Unreal Engine 源代码复制（或使用任何其他预定义的不允许的形式）。

## 5. License

**默认情况下**，本仓库采用 [AGPL-v3 License (or later)](vscode-file://vscode-app/c:/Users/Huang/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) 进行许可。其他要求参见 [English](./README.md) 的Readme
