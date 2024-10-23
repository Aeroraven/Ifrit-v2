import os
import subprocess

def compile_shader(shader):
    print(f"Compiling {shader}")
    subprocess.run(["glslc", shader, "-o", shader + ".spv"])

for file in os.listdir():
    if file.endswith(".vert") or file.endswith(".frag") or file.endswith(".comp"):
        compile_shader(file)
