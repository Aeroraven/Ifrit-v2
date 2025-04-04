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


# add executable
add_executable(ifrit.demo.syaro ifrit.demo.syaro.cpp)
add_executable(ifrit.demo.ayanami ifrit.demo.ayanami.cpp)
add_executable(ifrit.demo.test ifrit.demo.test.cpp)
add_executable(ifrit.demo.memdiag ifrit.demo.memdiag.cpp)

include_directories(${IFRIT_PROJECT_SUBDIR}/softrenderer/include)
include_directories(${IFRIT_PROJECT_SUBDIR}/display/include)


# add link libraries to a variable
set(IFRIT_DEMO_LINK_LIBRARIES ifrit.softrenderer ifrit.meshproclib ifrit.display glfw ifrit.vkrenderer ifrit.runtime ifrit.core)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(ifrit.demo.syaro meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES})
    target_link_libraries(ifrit.demo.ayanami meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES})
    target_link_libraries(ifrit.demo.test meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES})
    target_link_libraries(ifrit.demo.memdiag meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES})
else()
    target_link_libraries(ifrit.demo.syaro meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES} dl pthread)
    target_link_libraries(ifrit.demo.ayanami meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES} dl pthread)
    target_link_libraries(ifrit.demo.test meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES} dl pthread)
    target_link_libraries(ifrit.demo.memdiag meshoptimizer ${IFRIT_DEMO_LINK_LIBRARIES} dl pthread)
endif()
