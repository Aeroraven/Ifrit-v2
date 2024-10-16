cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Component-LLVMExec)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_STATIC_LIBRARY_PREFIX "")

# Set the output directory for the build executables and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin)
set(IFRIT_IRCOMPILE_OUT_DIR ${CMAKE_BINARY_DIR}../../bin)
add_library(ifrit.ircompile SHARED 
    ./src/LLVMExecRuntime.cpp
)

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
    set_target_properties(ifrit.ircompile PROPERTIES
        LINK_FLAGS "-Wl,--subsystem,windows,--output-def,../../../bin/ifrit.ircompile.def,--out-implib,ifrit.ircompile.a"
    )
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
        LINK_FLAGS "-Wl,--out-implib,ifrit.ircompile.a"
    )
    target_link_libraries(ifrit.ircompile PRIVATE LLVM-10)
endif()

set_target_properties(ifrit.ircompile PROPERTIES PREFIX "")
set_target_properties(ifrit.ircompile PROPERTIES
    OUTPUT_NAME "ifrit.ircompile"
)
# Print ircompile output directory
message(STATUS "[IFRIT/Output]: ${IFRIT_IRCOMPILE_OUT_DIR}")
# If MinGW
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # prebuild command: remove ifrit.ircompile*
        add_custom_command(TARGET ifrit.ircompile PRE_BUILD
            COMMAND cmd /c del ifrit.ircompile.*
            WORKING_DIRECTORY ${IFRIT_IRCOMPILE_OUT_DIR}
        )
        # libifrit.ircompile.dll.a rename to ifrit.ircompile.a
        add_custom_command(TARGET ifrit.ircompile POST_BUILD
            COMMAND cmd /c ren libifrit.ircompile.dll.a ifrit.ircompile.a
            WORKING_DIRECTORY ${IFRIT_IRCOMPILE_OUT_DIR}
        )
        add_custom_command(TARGET ifrit.ircompile POST_BUILD
            COMMAND dlltool --input-def ifrit.ircompile.def --dllname ifrit.ircompile.dll --output-lib ifrit.ircompile.lib
            WORKING_DIRECTORY ${IFRIT_IRCOMPILE_OUT_DIR}
        )
        add_custom_command(
            TARGET ifrit.ircompile POST_BUILD
            COMMAND lib /def:ifrit.ircompile.def /out:ifrit.ircompile.lib /machine:x64
            WORKING_DIRECTORY ${IFRIT_IRCOMPILE_OUT_DIR}
        )
    endif()
endif()