cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Component-LLVMExec)
set(CMAKE_CXX_STANDARD 14)

# Set the output directory for the build executables and libraries
set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/ProjectShared.cmake)
# Reset to CXX14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${IFRIT_PROJECT_DIR}/include)
add_library(ifrit.ircompile SHARED 
    ./src/LLVMExecRuntime.cpp
)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})
message(STATUS "[IFRIT/EnvCheck]: Working Directory ${IFRIT_PROJECT_DIR}")

if(NOT MSVC)
    target_compile_options(ifrit.ircompile PRIVATE -O2)
    target_compile_options(ifrit.ircompile PRIVATE
        -fno-exceptions
        -fno-rtti
    )
endif()
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++ -fno-stack-protector")
endif()

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
    
    if(MSVC)
        target_link_directories(ifrit.ircompile PRIVATE ${IFRIT_GLOBAL_LLVM_DIR_WINDOWS}/build/Debug/lib) #TODO
    else()
        target_link_directories(ifrit.ircompile PRIVATE ${IFRIT_GLOBAL_LLVM_DIR_WINDOWS}/build/lib)
    endif()
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
        psapi
        shell32
        ole32
        uuid
        advapi32
    )
    if(NOT MSVC)
        #add target link libraries: LLVMX86Utils
        target_link_libraries(ifrit.ircompile PRIVATE LLVMX86Utils z)
    endif()
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