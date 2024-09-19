# Ifrit-v2 

## Required Dependencies

### Env Requirement

OpenGL is required to run the project. Only `Ubuntu (WSL2)` and `Windows` are tested.

Other linux distributions might extra configurations to run the project (Too lazy to write `CMakeLists.txt` for other distributions).

### Compiler Toolchain

You have to install a compiler toolchain to build the project. The following compilers are tested and recommended.
- CMake 3.28
- Windows: 
  - MSVC 19.29 (Visual Studio 2022)
  - MinGW-w64 11.0 (GCC 10 is the minimum requirement)

- Ubuntu:
  - GCC 13.2 (GCC 10 is the minimum requirement)

#### External Headers
- **GLAD**: 
  1. Download from https://glad.dav1d.de/
  2. Generate OpenGL loader with `c/c++` language and `gl` version `3.3`
  3. Place `glad.c` in `core/include/dependency/GLAD/glad.c`
  4. Place `glad.h` in `core/include/dependency/GLAD/glad/glad.h`
  5. Place `khrplatform.h` in `core/include/dependency/GLAD/KHR/khrplatform.h`

- **SPIRV-Header**:
  1. Clone repository from https://www.github.com/KhronosGroup/SPIRV-Headers
  2. Place `include/spirv/unified1/GLSL.std.450.h` in `core/include/dependency/glsl.std.450`
  3. Place `include/spirv/unified1/spirv.hpp11` in `core/include/dependency/spirv.h`

- **SBT-Image**:
  1. Clone repository from https://www.github.com/nothings/stb
  2. Place `stb_image.h` in `core/include/dependency/stb_image.h`
 

### Dependencies
#### LLVM 10

Note that only LLVM-10 is tested. Unexpected behaviors are observed in LLVM-18 and LLVM-20 (compiled by `msvc` or `MinGW-w64`).

Follow the instructions to install LLVM (https://releases.llvm.org/download.html)

```bash
# For ubuntu
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 10
```

```bash
# Build from source (Windows)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/10.x
cmake -S llvm -B build -G "MinGW Makefiles" -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

#### GLFW 3.3
  
```bash
git clone https://github.com/glfw/glfw.git
cd glfw
cmake -S . -B build  -DCMAKE_BUILD_TYPE=Release
cd build
make
```



## Recommended Dependencies

### Compiler Toolchain

Only the following compilers are tested and recommended with CUDA integration.
- Windows: 
  - MSVC 19.29 (Visual Studio 2022)


### AVX2 Support

Check https://www.intel.com/content/www/us/en/support/articles/000090473/processors/intel-core-processors.html for AVX2 support.

### CUDA 12.6

Follow the instructions to install CUDA (https://developer.nvidia.com/cuda-downloads)

Note that if you are using CUDA, ensure version requirements are met. (At least CUDA 12.5. CUDA 12.4 will cause a compilation error on MSVC)

### .NET 8.0

You can simply install Visual Studio 2022 to get .NET 8.0.


## Build Configuration

### For Visual Studio 2022

1. Open the project with `Ifrit-v2.sln`
2. Change all library paths to your local path. (E.g. llvm and glfw3)
3. Build the project.

### For CMake

> No CUDA support is provided in current CMakeLists,txt. You have to manually add CUDA support in `CMakeLists.txt`.

1. Some library paths are hardcoded in `CMakeLists.txt`. Change them to your local path.
2. Run the following commands in the root directory of the project.
```bash
cmake -S . -B build 
cd build
make
```

## Run

# For Linux
```bash
export LD_LIBRARY_PATH=/path/to/Ifrit.Components.LLVMExec.so;$LD_LIBRARY_PATH
./core/bin/IfritMain
LIBGL_ALWAYS_SOFTWARE=1 ./core/bin/IfritMain # Maybe in WSL2
```

