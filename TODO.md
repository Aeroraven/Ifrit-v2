# Ifrit-v2 /  TODO 

## Bugs

- [ ] Syaro: no objects will got tested if first instance cull pass reject all objects using prev-frame hzb

  


## Artifacts

- [ ] Syaro: TAA flickering for distant tiny objects

- [ ] Syaro: LOD switching not smooth (confirming)
  - [x] <s>Abrupt attribute change (causing lighting changes when lod switches)</s>
  - [x] <s>Losing thin objects when switching to low-lod objects</s>
  
- [ ] Syaro: PCF shadow not smooth

  


## Performance

- [ ] Syaro: memory fragmentation and low memory utilization.
- [ ] Syaro: low thread utilization for small meshes (eg for almost no bvh nodes)
- [ ] Syaro: compute SW rasterizer got redundant computations in bbox iteration.
- [ ] Syaro: compute SW rasterizer got incorrect parallelism
    - [ ] Considering async compute
- [ ] Syaro: improper compute shader usage
    - Reference: https://computergraphics.stackexchange.com/questions/9956/performance-of-compute-shaders-vs-fragment-shaders-for-deferred-rendering

    

## Resolved

- Syaro: Alleviate abrupt attribute change between different mesh lods (250115)

  | After Fixing                                       | Before Fixing                                      |
  | -------------------------------------------------- | -------------------------------------------------- |
  | ![image-20250111153548338](docs/imgtodo/nlod1.png) | ![image-20250111153512009](docs/imgtodo/nlod2.png) |

- Syaro: Missing meshes, caused by wrong HiZ mip selection (250111)

  | After Fixing                                                 | Before Fixing                                                |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20250111153548338](docs/imgtodo/image-20250111153548338.png) | ![image-20250111153512009](docs/imgtodo/image-20250111153512009.png) |

- Syaro: Missing meshes, caused by wrong SW sampling location (250111)

  | After Fixing                                                 | Before Fixing                                                |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20250111144804794](docs/imgtodo/image-20250111144804794.png) | ![image-20250111144843767](docs/imgtodo/image-20250111144843767.png) |

  

- Syaro: FSR2 flickering & ghosting, when moving (250110)

- Syaro: incorrect startup (wrong view data for the first frame), referencing destroyed variable. (250103)

- Syaro: CSM uv overflow, incorrect bbox infinity init (250103)

- Syaro: GLTF node transform (250103)

- Syaro: CSM flickering while moving  

  