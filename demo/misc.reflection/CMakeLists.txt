# demo/artemis.mpm/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Get project root directory
set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR ${IFRIT_PROJECT_DIR_CORE})

# Include shared configuration
include(${IFRIT_PROJECT_DIR}/cmake/ProjectShared.cmake)
include(${IFRIT_PROJECT_DIR}/cmake/DemoTemplate.cmake)

# Create the demo project
ifrit_add_demo_project(misc.reflection)