cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Demo LANGUAGES CXX)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/buildinfra/ProjectShared.cmake)
include(${IFRIT_PROJECT_DIR}/projects/buildinfra/SIMDCheck.cmake)
include(${IFRIT_PROJECT_DIR}/projects/buildinfra/GLFW3Resolve.cmake)
include(${IFRIT_PROJECT_DIR}/projects/buildinfra/VulkanResolve.cmake)

# add shader path
add_definitions(-DIFRIT_DEMO_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/Assets/Scene")
add_definitions(-DIFRIT_DEMO_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/Assets/Shader")
add_definitions(-DIFRIT_DEMO_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/Assets/")
add_definitions(-DIFRIT_DEMO_CACHE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/Cache/")


# if(MSVC)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")
# endif()

file(GLOB IFRIT_DEMO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ifrit.demo.*.cpp")
set(IFRIT_DEMO_LIST_LOCAL)

foreach(SOURCE_FILE ${IFRIT_DEMO_SOURCES})
    string(REGEX REPLACE ".*/(ifrit.demo.*)\\.cpp" "\\1" SOURCE_NAME ${SOURCE_FILE})
    add_executable(${SOURCE_NAME} ${SOURCE_FILE})
    list(APPEND IFRIT_DEMO_LIST_LOCAL ${SOURCE_NAME})
endforeach()

#set ifrit_demo_list a global variable
set(IFRIT_DEMO_LIST ${IFRIT_DEMO_LIST_LOCAL} CACHE INTERNAL "List of demo stages" FORCE)

#print demo list
message(STATUS "[Ifrit.Build.Demo2] Found ${IFRIT_DEMO_LIST} demo stages.")
message(STATUS "[Ifrit.Build.Demo2] Found ${IFRIT_DEMO_LIST_LOCAL} demo stages.")

include_directories(${IFRIT_PROJECT_SUBDIR}/softrenderer/include)
include_directories(${IFRIT_PROJECT_SUBDIR}/display/include)


# add link libraries to a variable
set(IFRIT_DEMO_LINK_LIBRARIES 
    ifrit.softgraphics 
    ifrit.meshproclib 
    ifrit.display 
    glfw 
    ifrit.vkgraphics 
    ifrit.runtime 
    ifrit.core
    ${IFRIT_COMMON_DEPENDENCIES_SHARED})

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    foreach(TARGET_NAME ${IFRIT_DEMO_LIST_LOCAL})
        target_link_libraries(${TARGET_NAME} meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES})
    endforeach()
else()
    foreach(TARGET_NAME ${IFRIT_DEMO_LIST_LOCAL})
        target_link_libraries(${TARGET_NAME} meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES} dl pthread)
    endforeach()
endif()
    