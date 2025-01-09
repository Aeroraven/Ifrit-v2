# Ifrit-v2 /  TODO 

## Bugs

- [ ] Syaro: no objects will got tested if first instance cull pass reject all objects using prev-frame hzb
- [ ] Syaro: FSR2 flickering & ghosting, when moving


## Artifacts

- [ ] Syaro: TAA flickering for distant tiny objects
- [ ] Syaro: LOD switching not smooth
  - [ ] Abrupt attribute change (causing lighting changes when lod switches)

- [ ] Syaro: PCF shadow not smooth
- [ ] Syaro: Too aggressive culling


## Performance

- [ ] Syaro: memory fragmentation and low memory utilization.
- [ ] Syaro: low thread utilization for small meshes (eg for almost no bvh nodes)
- [ ] Syaro: compute SW rasterizer got redundant computations in bbox iteration.
- [ ] Syaro: compute SW rasterizer got incorrect parallelism
    - [ ] Considering async compute
- [ ] Syaro: improper compute shader usage
    - Reference: https://computergraphics.stackexchange.com/questions/9956/performance-of-compute-shaders-vs-fragment-shaders-for-deferred-rendering

    

## Resolved

- Syaro: incorrect startup (wrong view data for the first frame), referencing destroyed variable. (250103)

- Syaro: CSM uv overflow, incorrect bbox infinity init (250103)

- Syaro: GLTF node transform (250103)

- Syaro: CSM flickering while moving  

  