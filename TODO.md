# Ifrit-v2 /  TODO 

## Bugs

- [ ] Syaro: no objects will got tested if first instance cull pass reject all objects using prev-frame hzb
- [ ] Syaro: compute SW rasterizer losing depth precision when performing depth testing.
  - considering reverse z



## Artifacts

- [ ] Syaro: TAA flickering for distant tiny objects
- [x] Syaro: CSM flickering while moving  
- [ ] Syaro: incorect startup
- [ ] Syaro: LOD switching not smooth
- [ ] Syaro: CSM uv overflow



## Performance

- [ ] Syaro: memory fragmentation and low memory utilization.

- [ ] Syaro: low thread utilization for small meshes (eg for almost no bvh nodes)

- [ ] Syaro: compute SW rasterizer got redundant computations in bbox iteration.

- [ ] Syaro: compute SW rasterizer got incorrect parallelism

- [ ] Syaro: improper compute shader usage
    - Reference: https://computergraphics.stackexchange.com/questions/9956/performance-of-compute-shaders-vs-fragment-shaders-for-deferred-rendering

    