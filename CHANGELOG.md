# Ifrit-v2 /  TODO 

### Pending

- Ayanami: 非封闭网格(尤其是One-sided Mesh)的MDF存在较严重Artifact.
  - 可行参考：https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf
  - 可行缓解：(A) UDF + DF Expansion (B) Virtual Surface

- Ayanami: SDF BC4部分区域压缩后导致精度丢失Artifacts

  - 参考：https://jcgt.org/published/0011/03/06/paper-lowres.pdf

- Build: 编译时间过长，尤其是`slang`和`ifrit.runtime`

    

### Resolved

#### 2025.07

- Build: 部分重构和编译优化
  - 启用MSVC的多核编译（250717）
  - 减少无效头文件引用，添加前向声明
  - 移除在Logging头文件中的外部依赖spdlog暴露（250720）
  - 封装GUID，移除GUID头文件的外部依赖uuid暴露（250720）
  - 移除过度的序列化库cereal的头文件引用（250723）
  
- Core: AI辅助重构带来的低级错误，修复矩阵乘法（250702）

#### 2025.06

- VkGraphics: 部分兼容性调整，允许较低配置设备(如GTX1050)运行核心功能.

#### 2025.05

- Runtime: 修复HiZ的错误Layout导致RenderDoc的错误显示 (250530)

- Ayanami: 修复浮点误差导致的Surface Cache近平面丢失 (250529)

- Build: 降低编译耗时 (PCH, spdlog由head only转库文件，移除cereal冗余archive) (250501)

  | After Fixing       | Before Fixing   |
  | ------------------ | --------------- |
  | 编译用时: 4min 30s | 编译用时: 11min |

#### 2025.04

- Ayanami: 缓解 SDF RayMarching （包括离线Shadow Mask生成）的高延迟和内存带宽(Nsight Graphics: Long Scoreboard)：停止无效步进，优化DF Shadow Culling包围盒，调整Object Grid的Culling和Sort策略，自适应MDF体积，引入MDF的BC4压缩。(250412)

  | After Fixing                                                 | Before Fixing                                                |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | (包括强制GDF Object Grids/Direct Lighting Atlas重生成)<br/>帧用时: 84.1 ms (-80.6%)<br/><img src="docs/imgtodo/20250412051715.png" style="zoom:50%;" /> | (包括强制GDF Object Grids/Direct Lighting Atlas重生成)<br/>帧用时: 433.1 ms<br/><img src="docs/imgtodo/2025-04-12 051343.png" style="zoom:50%;" /> |
  | (Syaro GBuffer/CSM+完全的Direct Lighting Atlas重生成+Obj Grid 的Albedo Atlas可视化)<br/>帧用时: 19.51 ms<br/>(2501413) | (Syaro GBuffer/CSM+完全的Direct Lighting Atlas重生成+Obj Grid 的Albedo Atlas可视化)<br/>帧用时:  待测试 <br/> |

  

- Ayanami: 缓解 DF Shadow Culling 的错误和重复问题 (250407)

  | After Fixing                                                 | Before Fixing                                                |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="docs/imgtodo/2025-04-07 095430.png" style="zoom:50%;" /> | <img src="docs/imgtodo/2025-04-07 093307.png" style="zoom:60%;" /> |

    

- VkGraphics: 修复AllocateCommandBuffer导致的内存泄漏 (250404)

  - 参考：https://developer.download.nvidia.com/gameworks/events/GDC2016/Vulkan_Essentials_GDC16_tlorach.pdf#page=25.00


#### 2025.03

- RHI: RHI资源引用计数和待删除队列 (250325)

- Syaro: 修复错误的Cluster Group Culling (250324)

#### 2025.01
- Syaro: 缓解LoD切换时的顶点属性突变 (250115)

  | After Fixing                                       | Before Fixing                                      |
  | -------------------------------------------------- | -------------------------------------------------- |
  | ![image-20250111153548338](docs/imgtodo/nlod1.png) | ![image-20250111153512009](docs/imgtodo/nlod2.png) |

- Syaro: 由HiZ采样位置错误导致的图元丢失 (250111)

  | After Fixing                                                 | Before Fixing                                                |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20250111153548338](docs/imgtodo/image-20250111153548338.png) | ![image-20250111153512009](docs/imgtodo/image-20250111153512009.png) |

- Syaro: 由软渲染采样位置错误导致的图元丢失 (250111)

  | After Fixing                                                 | Before Fixing                                                |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20250111144804794](docs/imgtodo/image-20250111144804794.png) | ![image-20250111144843767](docs/imgtodo/image-20250111144843767.png) |

  

- Syaro: FSR2 移动时的闪烁和拖影 (250110)

- Syaro: 由引用已析构的对象造成的第一帧的View信息无效. (250103)

- Syaro: CSM UV 溢出, 导致错误的AABB初始化 (250103)

- Syaro: GLTF 错误的节点变换 (250103)

- Syaro: CSM 移动时闪烁

  