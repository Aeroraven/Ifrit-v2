# Ifrit-v2 /  TODO 

### Pending

- Ayanami: 非封闭网格(尤其是One-sided Mesh)的MDF存在较严重Artifact.
  - 可行参考：https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf
  - 可行缓解：(A) UDF + DF Expansion (B) Virtual Surface



### Resolved

#### 2025.04

- Ayanami: 缓解 DF Shadow Culling 的错误和重复问题 (250407)

  | After Fixing                                                 | Before Fixing                           |
  | ------------------------------------------------------------ | --------------------------------------- |
  | <img src="docs/imgtodo/2025-04-07 095430.png" style="zoom:80%;" /> | ![](docs/imgtodo/2025-04-07 093307.png) |

    

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

  