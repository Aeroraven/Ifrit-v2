cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Component-LLVMExec)
set(CMAKE_CXX_STANDARD 14)

# Set the output directory for the build executables and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin)
set(IFRIT_IRCOMPILE_OUT_DIR ${CMAKE_BINARY_DIR}../../bin)
add_library(ifrit.ircompile SHARED 
    ./src/LLVMExecRuntime.cpp
)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})
message(STATUS "[IFRIT/EnvCheck]: Working Directory ${IFRIT_PROJECT_DIR}")

target_compile_options(ifrit.ircompile PRIVATE -O3)
target_compile_options(ifrit.ircompile PRIVATE
    -fno-exceptions
    -fno-rtti
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++ -fno-stack-protector")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(ifrit.ircompile PRIVATE
        IFRIT_COMPONENT_LLVMEXEC_EXPORT
        _FILE_OFFSET_BITS=64
        __STDC_CONSTANT_MACROS
        __STDC_FORMAT_MACROS
        __STDC_LIMIT_MACROS
    )
    include_directories(
       ${IFRIT_GLOBAL_LLVM_DIR_WINDOWS}/llvm/include
       ${IFRIT_GLOBAL_LLVM_DIR_WINDOWS}/build/include
    )
    target_link_directories(ifrit.ircompile PRIVATE ${IFRIT_GLOBAL_LLVM_DIR_WINDOWS}/build/lib)
    target_link_libraries(ifrit.ircompile PRIVATE
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
else()
    target_compile_definitions(ifrit.ircompile PRIVATE
        IFRIT_COMPONENT_LLVMEXEC_EXPORT
        _FILE_OFFSET_BITS=64
        _GNU_SOURCE
        __STDC_CONSTANT_MACROS
        __STDC_FORMAT_MACROS
        __STDC_LIMIT_MACROS
    )
    include_directories(${IFRIT_GLOBAL_LLVM_DIR_LINUX}/include)
    target_link_directories(ifrit.ircompile PRIVATE ${IFRIT_GLOBAL_LLVM_DIR_LINUX}/lib)
    set_target_properties(ifrit.ircompile PROPERTIES
        LINK_FLAGS "-Wl,--out-implib,libifrit.ircompile.a"
    )
    target_link_libraries(ifrit.ircompile PRIVATE LLVM-10)
endif()

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/WindowsLibConv.cmake)
WindowsPrebuild(ifrit.ircompile ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.ircompile")
WindowsPostbuild(ifrit.ircompile "libifrit.ircompile.def" "libifrit.ircompile.dll" "libifrit.ircompile.lib" ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "x64")
AddImpLib(ifrit.ircompile ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.ircompile")