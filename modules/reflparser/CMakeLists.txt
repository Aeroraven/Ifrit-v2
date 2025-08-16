cmake_minimum_required(VERSION 3.20)
project(ifrit.reflparser)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow user to specify custom clang include directory and root
set(CLANG_INCLUDE_DIR "" CACHE STRING "Custom path to clang-c headers")
set(CUSTOM_CLANG_ROOT "" CACHE STRING "Custom path to clang root directory")
set(ADDITIONAL_INCLUDE_DIRS "" CACHE STRING "Additional include directories for the project")
set(CUSTOM_BIN_DIR "" CACHE STRING "Custom binary output directory")

if(CUSTOM_BIN_DIR)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CUSTOM_BIN_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CUSTOM_BIN_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CUSTOM_BIN_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CUSTOM_BIN_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CUSTOM_BIN_DIR})
    message(STATUS "Using custom binary output directory: ${CUSTOM_BIN_DIR}")
endif()

# Find Visual Studio installation or use custom clang root
if(WIN32)
    # Use custom clang root if provided
    if(CUSTOM_CLANG_ROOT)
        set(LIBCLANG_ROOT "${CUSTOM_CLANG_ROOT}")
        message(STATUS "Using custom clang root: ${LIBCLANG_ROOT}")
    else()
        # Common Visual Studio installation paths
        set(VS_PATHS
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise"
            "C:/Program Files/Microsoft Visual Studio/2022/Professional"
            "C:/Program Files/Microsoft Visual Studio/2022/Community"
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise"
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional"
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community"
        )
        
        foreach(VS_PATH ${VS_PATHS})
            if(EXISTS "${VS_PATH}")
                set(VISUAL_STUDIO_PATH "${VS_PATH}")
                break()
            endif()
        endforeach()
        
        if(NOT VISUAL_STUDIO_PATH)
            message(FATAL_ERROR "Visual Studio installation not found")
        endif()
        
        # Set libclang paths
        set(LIBCLANG_ROOT "${VISUAL_STUDIO_PATH}/VC/Tools/Llvm/x64")
        message(STATUS "Found Visual Studio at: ${VISUAL_STUDIO_PATH}")
        message(STATUS "Using libclang from: ${LIBCLANG_ROOT}")
    endif()
    
    # Use custom include directory if provided, otherwise use default
    if(CLANG_INCLUDE_DIR)
        set(LIBCLANG_INCLUDE_DIR "${CLANG_INCLUDE_DIR}/clang-c")
        set(LIBCLANG_INCLUDE_DIR_UPPER "${CLANG_INCLUDE_DIR}/")
        include_directories(${LIBCLANG_INCLUDE_DIR_UPPER})
        message(STATUS "Using custom clang include directory: ${LIBCLANG_INCLUDE_DIR}")
    else()
        set(LIBCLANG_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/dependencies/clang-c")
        message(STATUS "Using default clang include directory: ${LIBCLANG_INCLUDE_DIR}")
    endif()
    
    set(LIBCLANG_LIBRARY "${LIBCLANG_ROOT}/lib/libclang.lib")
    set(LIBCLANG_DLL "${LIBCLANG_ROOT}/bin/libclang.dll")
    
    # Verify paths exist
    if(NOT EXISTS "${LIBCLANG_INCLUDE_DIR}")
        message(FATAL_ERROR "libclang headers not found at ${LIBCLANG_INCLUDE_DIR}")
    endif()
    
    if(NOT EXISTS "${LIBCLANG_LIBRARY}")
        message(FATAL_ERROR "libclang library not found at ${LIBCLANG_LIBRARY}")
    endif()
    
    if(NOT EXISTS "${LIBCLANG_DLL}")
        message(FATAL_ERROR "libclang DLL not found at ${LIBCLANG_DLL}")
    endif()
endif()

# Collect all source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.c")
file(GLOB_RECURSE HEADERS "src/*.h" "src/*.hpp")

# Extra definitions
# DIFRIT_COMMON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include
# DIFRIT_COMMON_DEPS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/dependencies
add_definitions(-DIFRIT_COMMON_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/include")
add_definitions(-DIFRIT_COMMON_DEPS_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/dependencies")
add_definitions(-DIFRIT_GATHER_HEADS_OUT="${CMAKE_SOURCE_DIR}/modules/reflparser/auto_generated/Intermediate")
add_definitions(-DIFRIT_META_OUT="${CMAKE_SOURCE_DIR}/include/ifrit.generated/reflparser.gen/ProjectReflMetaData.cpp")
if(ADDITIONAL_INCLUDE_DIRS)
    include_directories(${ADDITIONAL_INCLUDE_DIRS})
    message(STATUS "Adding additional include directories: ${ADDITIONAL_INCLUDE_DIRS}")
endif()
# Create executable
add_executable(ifrit.reflparser ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(ifrit.reflparser PRIVATE 
    ${LIBCLANG_INCLUDE_DIR}
    "${CMAKE_SOURCE_DIR}/dependencies"
    "${CMAKE_SOURCE_DIR}/include"
    src
)

# Link with libclang
target_link_libraries(ifrit.reflparser PRIVATE ${LIBCLANG_LIBRARY})

# Add compile definitions for Windows
if(WIN32)
    target_compile_definitions(ifrit.reflparser PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
    )
endif()

# Set output directory
if(CUSTOM_BIN_DIR)
    set_target_properties(ifrit.reflparser PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CUSTOM_BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CUSTOM_BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CUSTOM_BIN_DIR}"
    )
else()
    set_target_properties(ifrit.reflparser PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Copy libclang.dll to output directory for runtime
add_custom_command(TARGET ifrit.reflparser POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${LIBCLANG_DLL}"
    "$<TARGET_FILE_DIR:ifrit.reflparser>"
    COMMENT "Copying libclang.dll to output directory"
)