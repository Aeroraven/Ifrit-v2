cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Component-LLVMExec)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_STATIC_LIBRARY_PREFIX "")
add_library(Ifrit.Components.LLVMExec SHARED 
    ./src/LLVMExecRuntime.cpp
)
include_directories(
    C:/llvm-project/llvm/include
    C:/llvm-project/build/include
)
target_compile_options(Ifrit.Components.LLVMExec PRIVATE -O3)
target_compile_definitions(Ifrit.Components.LLVMExec PRIVATE
    IFRIT_COMPONENT_LLVMEXEC_EXPORT
    _FILE_OFFSET_BITS=64
    __STDC_CONSTANT_MACROS
    __STDC_FORMAT_MACROS
    __STDC_LIMIT_MACROS
)
target_compile_options(Ifrit.Components.LLVMExec PRIVATE
    -fno-exceptions
    -fno-rtti
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
target_link_libraries(Ifrit.Components.LLVMExec PRIVATE
    LLVMOrcJIT
    LLVMPasses
    LLVMipo
    LLVMInstrumentation
    LLVMVectorize
    LLVMLinker
    LLVMIRReader
    LLVMAsmParser
    LLVMOrcError
    LLVMJITLink
    LLVMInterpreter
    LLVMX86Disassembler
    LLVMX86AsmParser
    LLVMX86CodeGen
    LLVMCFGuard
    LLVMGlobalISel
    LLVMSelectionDAG
    LLVMAsmPrinter
    LLVMDebugInfoDWARF
    LLVMCodeGen
    LLVMScalarOpts
    LLVMInstCombine
    LLVMAggressiveInstCombine
    LLVMTransformUtils
    LLVMBitWriter
    LLVMX86Desc
    LLVMMCDisassembler
    LLVMX86Utils
    LLVMX86Info
    LLVMMCJIT
    LLVMExecutionEngine
    LLVMTarget
    LLVMAnalysis
    LLVMProfileData
    LLVMRuntimeDyld
    LLVMObject
    LLVMTextAPI
    LLVMMCParser
    LLVMBitReader
    LLVMMC
    LLVMDebugInfoCodeView
    LLVMDebugInfoMSF
    LLVMCore
    LLVMRemarks
    LLVMBitstreamReader
    LLVMBinaryFormat
    LLVMSupport
    LLVMDemangle
    z
    psapi
    shell32
    ole32
    uuid
    advapi32
)
target_link_directories(Ifrit.Components.LLVMExec PRIVATE C:/llvm-project/build/lib)
set_target_properties(Ifrit.Components.LLVMExec PROPERTIES
    LINK_FLAGS "-Wl,--subsystem,windows,--output-def,Ifrit.Components.LLVMExec.def,--out-implib,Ifrit.Components.LLVMExec.a"
)

set_target_properties(Ifrit.Components.LLVMExec PROPERTIES PREFIX "")
set_target_properties(Ifrit.Components.LLVMExec PROPERTIES
    OUTPUT_NAME "Ifrit.Components.LLVMExec"
)
# If MinGW
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # libIfrit.Components.LLVMExec.dll.a rename to Ifrit.Components.LLVMExec.a
        add_custom_command(TARGET Ifrit.Components.LLVMExec POST_BUILD
            COMMAND cmd /c ren libIfrit.Components.LLVMExec.dll.a Ifrit.Components.LLVMExec.a
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_custom_command(TARGET Ifrit.Components.LLVMExec POST_BUILD
            COMMAND dlltool --input-def Ifrit.Components.LLVMExec.def --dllname Ifrit.Components.LLVMExec.dll --output-lib Ifrit.Components.LLVMExec.lib
        )
        add_custom_command(
            TARGET Ifrit.Components.LLVMExec POST_BUILD
            COMMAND lib /def:Ifrit.Components.LLVMExec.def /out:Ifrit.Components.LLVMExec.lib /machine:x64
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
endif()