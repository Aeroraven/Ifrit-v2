Test performed on 2048x2048 RGBA FP32 Image + 2048x2048 FP32 Depth Attachment. Time consumption in presentation stage (displaying texture via OpenGL) is ignored. Note that some triangles might be culled or clipped in the pipeline. 

Test were performed before git commit `7e6c34ad836842c02fcc9aa7dc89d5d01cd6cb66`.

| Model             | Triangles | CPU Single Thread | CPU Multi Threads | CUDA w/ Copy-back | CUDA w/o Copy-back |
| ----------------- | --------- | ----------------- | ----------------- | ----------------- | ------------------ |
| Naganohara Yomiya | 70275     | 38                | 80                | 123               | 2857               |
| Stanford Bunny    | 208353    | 20                | 80                | 124               | 2272               |
| Khronos Sponza    | 786801    | 2                 | 10                | 125               | 500                |
| Intel Sponza      | 11241912  | 1                 | 7                 | 125               | 198                |
| Upper Bound       | /         | 144 ※ (Memset)    | 259 ※ (Memset)    | 125 ※ (PCIe)      | ∞                  |