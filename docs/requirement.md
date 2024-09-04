# Ifrit-v2 

## Dependencies

### Ifrit-v2/Core

#### Minimal Requirement

- **Component Dependency**:

  - Ifrit-v2/Components/LLVMExec

- **Display Dependencies**: 

  - OpenGL (GLFW3.3 + GLAD)

- **Compilation Dependencies:** One of following environments. Requires `c++20` support.

  - MSVC 19.29 + Visual Studio 2022 
  - CMake 3.28 + GCC 13.2 (MinGW Included) `[WARNING: Not tested on recent commits]`

- **Optional**: CUDA is optional, but when compiling with CUDA:

  - CUDA >= 12.5

    

#### Recommended Requirement

- **Hardware Requirements:**  
  - CUDA 12.6
  - AVX2 Support
- **Display Dependencies**: 
  - OpenGL (GLFW3.3 + GLAD)
- **Compilation Dependencies:** Requires `c++20` support.
  - MSVC 19.29 + Visual Studio 2022 



### Ifrit-v2/Components/LLVMExec

#### Minimal Requirement 

- **Library Dependencies:**
  - LLVM 10 (>=10.0.0, <11.0.0)
- **Compilation Dependencies:** Both
  - MinGW-w64 11.0 (or equivalent compiler)
  - MSVC 19.29 (Library Manager Version 14.29)



### Ifrit-v2/Wrapper/CSharp

#### Minimal Requirement 

- **Dependency**: Ifrit-v2/Core
- **Framework**: .NET 8.0





## Setup

### Dependency Installation

Some dependencies should be prepared before compiling.

- Place `GLAD` dependency in ` core/include/dependency/GLAD/glad/glad.h` and `core/include/dependency/GLAD/KHR/khrplatform.h`
- Place `sbt_image` in `core/include/dependency/sbt_image.h`
  - Place `spirv-header` in `core/include/dependency/spirv.h`


Change CUDA path and GLFW3 library path in `core/CMakeLists.txt` 



### Ifrit-v2/Core

#### Compile using G++ / MinGW

Follow instructions to build

```cmake
cmake -S core -B ./build
cd build
make
```



#### Compile using Visual Studio

Open `Ifrit-v2x.sln` in Visual Studio 2022.

- Edit the property sheet to help the linker find CUDA and GLFW3 library file.



### Ifrit-v2/Components/LLVMExec

Run build script in `components/llvmexec/Ifrit-v2-Component-LLVMExec`



### Ifrit-v2/Wrapper/CSharp

Open `Ifrit-v2x.sln` in Visual Studio 2022.