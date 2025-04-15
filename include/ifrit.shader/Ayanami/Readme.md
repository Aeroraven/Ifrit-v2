## Ifrit-v2 / Ayanami / Shaders

### 1. Basics

- **Radiant Flux**: Energy emitted/reflected/transmitted per time.
  $$
  \Phi=\frac{\mathrm{d}Q}{\mathrm{d}t}
  $$

- **Radiant Intensity**: Power per solid angle emitted by a point light source
  $$
  I(w)=\frac{\mathrm{d}\Phi}{\mathrm{d}w}
  $$

- **Irradiance**: Power per unit area on surface point

- **Radiosity**: Power per unit area leaving surface (integral of exiting radiance)
  $$
  E(x)=\frac{\mathrm{d}\Phi(x)}{\mathrm{d}A}
  $$

- **Radiance**: Power emitted/reflected/transmitted per solid angle and per area unit.
  $$
  L(p,w)=\frac{\mathrm{d}\Phi(p,w)}{\mathrm{d}w\mathrm{d}A\cos\theta}
  $$

- **Incident/Exiting Radiance**:
  $$
  L(p,w)=\frac{\mathrm{d}E(p)}{\mathrm{d}w\cos\theta}\\
  L(p,w)=\frac{\mathrm{d}I(p,w)}{\mathrm{d}A\cos\theta}\\
  $$

- **Rendering Equation**:
  $$
  L_o(p,w)=L_e(p,w)+\int_{H^2}f(p,w_i\to w_o)L_i(p,w_i)(n\cdot w_i)\mathrm{d}w
  $$

- **Spherical Harmonics**: To sample an accumulated SH, where $Y_{l,m}$ is the basis function.
  $$
  f(r)=\sum_{l=0}^{b}\sum_{m=-l}^lC_{l,m}Y_{l,m}(r)
  $$
  Then, due to orthogonal basis
  $$
  \int_Sf(s)Y_{l',m'}(s)\mathrm{ds}=\int_S\sum_{l=0}^{b}\sum_{m=-l}^lC_{l,m}Y_{l,m}(s)Y_{l',m'}(s)\mathrm{ds}=\int_SC_{l',m'}Y_{l',m'}(s)Y_{l',m'}(s)\mathrm{d}s = C_{l',m'}
  $$
  With monte carlo estimator
  $$
  C_{l,m}=\frac{1}{N}\sum_{i=1}^N\frac{f(s)Y_{l,m}(s)}{p(s)}
  $$

- **Cosine Lobe (2-Band SH)** : Project $\max(w\cdot R,0)$ onto SH
  
  - $l=0,m=0$, $Y=\frac{1}{2\sqrt\pi}$. Then, we have
    $$
    Y'=\int_H \frac{\cos \phi\sin\phi}{2\sqrt\pi}\mathrm{d}\phi\mathrm{d}\theta=\frac{1}{2\sqrt\pi}(2\pi)\frac{1}{4}\int_0^\pi\sin\phi\mathrm{d}\phi=\sqrt{\frac{\pi}{2}}
    $$
  
  - $l=1,Y=\sqrt{\frac{3}{4\pi}}w$. Consider the integral
    $$
    I(\vec{n})=\int_{H(\vec{n})}\sqrt\frac{3}{4\pi}\vec{w} (\vec{w}\cdot\vec{n}) \mathrm{d}\vec{w}\\
    H(\vec{n}):\{\vec{p}(x,y,z)|x^2+y^2+z^2=1,\vec{p}\cdot\vec{n}\geq0\}
    $$
    With special case, we have 
    $$
    I_z=2\pi\int_0^{\pi/2}(\cos\phi\sin\phi) \sin\phi\mathrm{d}\phi\\
    =2\pi\int_0^{\pi/2}\cos\phi(1-\cos^2\phi)\mathrm{d}\phi\\
    =2\pi\int_0^{\pi/2}\cos\phi\mathrm{d}\phi-2\pi\int_0^{\pi/2}\cos^3\phi\mathrm{d}{\phi}
    $$
    Using Wallis' integral
    $$
    I_z=2\pi(1-\frac{2}{3})=\frac{2\pi}{3}
    $$
    Then, considering the rotation, the coefficient will be 
    $$
    I(\vec{n})= I_z\sqrt\frac{3}{4\pi}\vec{n}=\sqrt{\frac{\pi}{3}}\vec{n}
    $$
  
  
  
- **Sampling on Sphere**: 
  $$
  \mathrm{d}A=\sin\theta\mathrm{d}\theta\mathrm{d}\phi
  $$
  

​	Then, the sampling probability $P(A)=P(\theta)P(\phi)\sin\theta\mathrm{d}\theta\mathrm{d}\phi$. Then, the MDF can be 
$$
F(\theta)=\int_0^\theta P(\theta)\int_0^\pi P(\phi)\sin\phi\mathrm{d}\phi\mathrm{d}\theta \\
F(\phi)=\int_0^\phi P(\phi)\sin\phi\int_0^{2\pi}P(\theta)\mathrm{d}\theta\mathrm{d}\phi
$$

​	For uniform sampling, we want $P(A)=\frac{1}{4\pi}\mathrm{d}A=\frac{\sin\phi}{4\pi}\mathrm{d}\theta\mathrm{d}\phi$. Then we have
$$
F(\theta)=\int_0^\theta\int_0^{\pi}\frac{\sin\phi}{4\pi}\mathrm{d}\phi\mathrm{d}\theta=\frac{1}{2\pi}\theta
$$

$$
F(\phi)=\int_0^\phi\int_0^{2\pi}\frac{\sin\phi}{4\pi}\mathrm{d}\theta\mathrm{d}\phi=\frac{1-\cos\phi}{2}
$$

​	To map $(u,v)\sim U(0,1)^2$ to PDF for $f(\theta)$, $f(\phi)$,  we apply inverse sampling transform
$$
\theta=\arccos(1-2u),\quad \cos\theta=1-2u\\
\phi=2\pi v
$$
​	And cosine sampling on hemisphere, we want $P(A)=\frac{k\cos\phi}{2\pi}\mathrm{d}A=\frac{k\cos\phi\sin\phi}{2\pi}\mathrm{d}\phi\mathrm{d}\theta$

​	Then, the total surface is 
$$
S=\int_0^\frac{\pi}{2}k\cos\phi\sin\phi\mathrm{d}\phi=\frac{1}{4}\int_0^\pi k\sin\phi\mathrm{d}\phi=\frac{1}{4}(-cos\pi+cos0)k=\frac{1}{2}k=1
$$
​	So,  we have
$$
P(A)=\frac{1}{\pi}\cos\phi\sin\phi\mathrm{d}\phi\mathrm{d}\theta
$$
​	And the MDFs
$$
F(\theta)=\int_0^\theta\int_0^\pi\frac{1}{4\pi}\sin\phi\mathrm{d}\phi\mathrm{d}\theta=\frac{1}{2\pi}\theta \\
F(\phi)=\int_0^\phi2\sin\phi\cos\phi\mathrm{d}\phi=\frac{1}{2}\left(1-\cos2\phi\right)=\frac{1}{2}\left(1-1+2\sin^2\phi\right)=\sin^2\phi=1-\cos^2\phi
$$
​	And use  inverse sampling transform
$$
\theta=2\pi u\\
\phi=\arccos\sqrt{1-v},\quad \cos\phi=\sqrt{1-v}
$$

### 2. Simplified Version for Lumen's Radiosity Trace

For each card tile, the tracer places $N$ probes, each has $K$ tracing rays. Let $P_{n,k}$ for the n-th probe and k-th ray. A probe is responsible for a subarea of a card tile (maybe calling it sub-tile)

- **Trace Center**: the trace ray's center for probe $n$ is randomly chosen inside sub-tile, and remains the same among all sub rays. Then tracer samples the card atlas for the probe's world position $W_{n}$

- **Trace Direction**: the tracer transforms $k$ into the tracing coordinate $T(k)=(Tx(k)+\epsilon_x,Ty(k)+\epsilon_y)$, with normalized uv range $[0,1]$. Different strategies are applied to map $T(k)\sim~U(0,1)^2$ into direction vector $V(k)$. Uniform or cosine transform are used to mapping vectors.

- **Field Tracing**: Start the Global DF tracing. After founding hit on GDF, it finds at most 4 objects in the object grid (the legacy voxel lighting is dropped) and evaluate the mesh card hit based on normal coincidence (squared), and hit points' normalized depth difference. Sample weight is calculated base on normal and depth difference. And the final lighting data is added into accumulator.

  - Writing into atlas `RWTraceHitDistanceAtlas`

  - Structure for threads: 

    ```
    (64,1,1) =>
    Card Tile: 8x8 => 1 Thread Group = 1 Tile
    Probe Range: 4x4 => 1 Tile == 4 Probe Ranges && 1 Probe = 16 Traces
    RadiosityTileSize = CARD_TILE_SIZE(8) / ProbeSpacingInRadiosityTexels(4) = 2
    
    Probe Id = TID / Traces Per Probe(16);
    CardTileIndex = Probe / Probes Per Tile (4);
    LinearIndexInCardTile = {1,2,3,4} => ProbeCoord = (0,0),(0,1),(1,0),(1,1)
    CoordInCardTile = (0,0)~(k,k), k = ProbeSpacingInRadiosityTexels(4)
    TraceTexelCoord = (0,0)~(p,p), p = HemisphereProbeResolution(4)
    
    Probe Atlas Coord: (Each probe range shares the same start)
    
    
    ```

    

- **Spatial Filter (TODO)** : For each probe $n$ and ray $k$, it gathers neighbor probe (within neighbor 4 coordinates, limited to the same card and valid probe), calculate the world position and normal using the method (tracing center) above. Filter weight is calculated using coplanarity and visibility.

  - Coplanarity: plane distance related to probe center distance. binarize.
  - The radiance resides in neighbor probe with **same Trace Texel Coord** is added into accumulator

- **Spherical Harmonics Transform (TODO)**: For each probe, calculate the 2-band SH coefficients. Each probe got 1 texel on 3 different atlases (r,g,b). Each stores the irradiance 2-band SH for R/G/B channel.
  $$
  C_k=\frac{1}{V({r_i})}\sum_{i=0}^{15}\frac{Y_k(r_i)}{P(r_i)}
  $$

- **Spherical Harmonics Integration (TODO)**: Each thread group processes one radiosity texel. It finds the nearest 4 probes and makes a weighted SH. Then, each texel calculates the radiance from the weighted irradiance SH. The radiosity is the exiting radiance, thus the final radiance can be:
  $$
  B(p)=\int_\Omega L(x\to w)\max(n\cdot w,0)\mathrm{d}w
  $$
  







