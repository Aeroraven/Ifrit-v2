# Ifrit-v2 /  TODO 


## Refactoring

- [ ] CMake: 跨环境兼容性
  - [ ] IRCompile: 移除硬编码LLVM地址

## Bugs

- [ ] Syaro: 如果第一次Instance Cull拒绝了所有的instance，导致后续Kernel不执行
- [ ] Syaro: Nanite's 的BVH错误实现
- [ ] Syaro: 缓存文件夹未正确创建 (+20250311)


## Artifacts
- [ ] Syaro: FSR2 移动时闪烁（错误的锐化效果）
- [ ] Syaro: SSGI Denoising 和 TAA/FSR2 不兼容 
- [ ] Syaro: TAA 远景小物体闪烁
- [x] Syaro: Mesh LOD 的切换不平滑
  - [x] <s>Abrupt attribute change (causing lighting changes when lod switches)</s>
  - [x] <s>Losing thin objects when switching to low-lod objects</s>
- [ ] Syaro: PCF 阴影不平滑



## Performance

- [ ] Syaro: 内存碎片化和内存复用.
- [ ] Syaro: 小型Mesh的Kernel效率低下
- [ ] Syaro: compute SW rasterizer 冗余循环
- [ ] Syaro: compute SW rasterizer 没有正确并行化
    - [ ] Considering async compute
- [ ] Syaro: 不合理的Compute Shader使用
    - Reference: https://computergraphics.stackexchange.com/questions/9956/performance-of-compute-shaders-vs-fragment-shaders-for-deferred-rendering


## Resolved


### 2025.01
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

  