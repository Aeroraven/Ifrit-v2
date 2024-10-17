cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-VkRenderer LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})
message(STATUS "[IFRIT/EnvCheck]: Working Directory ${IFRIT_PROJECT_DIR}")

set(IFRIT_PROJECT_SUBDIR "${IFRIT_PROJECT_DIR}/projects")

