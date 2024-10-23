import os
import subprocess

def compile_shader(shader):
    print(f"Compiling {shader}")
    subprocess.run(["glslc", shader, "-o", shader + ".spv"])

def compile_mesh_shader(shader):
    # --target-env=vulkan1.2 --target-spv=spv1.5
    print(f"Compiling {shader}")
    subprocess.run(["glslc", shader, "-o", shader + ".spv", "--target-env=vulkan1.2", "--target-spv=spv1.5"])


for file in os.listdir():
    if file.endswith(".vert") or file.endswith(".frag") or file.endswith(".comp"):
        compile_shader(file)
    if file.endswith(".mesh"):
        compile_mesh_shader(file)
