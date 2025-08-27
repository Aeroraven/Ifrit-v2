function(ifrit_add_demo_project DEMO_NAME)
    set(options CONSOLE_APP)
    set(oneValueArgs TARGET_NAME SOURCE_DIR ASSETS_DIR)
    set(multiValueArgs SOURCES HEADERS EXTRA_LIBRARIES EXTRA_DEPENDENCIES DEFINITIONS)
    cmake_parse_arguments(DEMO "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Set defaults
    if(NOT DEMO_TARGET_NAME)
        set(DEMO_TARGET_NAME "ifrit.demo.${DEMO_NAME}")
        message(STATUS "[Ifrit.Demo] Using default target name: ${DEMO_TARGET_NAME}")
    endif()
    
    if(NOT DEMO_SOURCE_DIR)
        set(DEMO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
    endif()
    
    if(NOT DEMO_ASSETS_DIR)
        set(DEMO_ASSETS_DIR "${CMAKE_SOURCE_DIR}/demo/shared/Content")
    endif()
    
    # Collect source files
    file(GLOB_RECURSE DEMO_SOURCES 
        "${DEMO_SOURCE_DIR}/*.cpp" 
        "${DEMO_SOURCE_DIR}/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/include.generated/*.cpp")
    
    if(NOT DEMO_HEADERS)
        file(GLOB_RECURSE DEMO_HEADERS 
             "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
             "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
    endif()
    
    # Add common definitions
    set(DEMO_COMMON_DEFINITIONS
        -DIFRIT_DEMO_SCENE_PATH="${DEMO_ASSETS_DIR}/Scene"
        -DIFRIT_DEMO_SHADER_PATH="${DEMO_ASSETS_DIR}/Shader"
        -DIFRIT_DEMO_ASSET_PATH="${DEMO_ASSETS_DIR}/"
        -DIFRIT_DEMO_CACHE_PATH="${CMAKE_SOURCE_DIR}/demo/shared/Cache/"
        -DIFRIT_DLL
        ${DEMO_DEFINITIONS}
    )
    
    # Create executable
    add_executable(${DEMO_TARGET_NAME} ${DEMO_SOURCES} ${DEMO_HEADERS})
    
    # Set properties
    target_compile_definitions(${DEMO_TARGET_NAME} PRIVATE ${DEMO_COMMON_DEFINITIONS})
    target_include_directories(${DEMO_TARGET_NAME} PRIVATE 
        ${IFRIT_PROJECT_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include.generated
    )
    
    # Link libraries
    target_link_libraries(${DEMO_TARGET_NAME} 
        ${IFRIT_DEMO_LINK_LIBRARIES}
        ${DEMO_EXTRA_LIBRARIES}
    )
    
    # Add dependencies
    add_dependencies(${DEMO_TARGET_NAME} 
        ${IFRIT_DEMO_DEPENDENCIES}
        ${DEMO_EXTRA_DEPENDENCIES}
    )
    
    # Set folder in IDE
    set(IFRIT_GROUP_DEMO "Project Ifrit/Demo")
    set(IFRIT_GROUP_DEMO_GEN "Project Ifrit/Demo.Generated")
    if(IFRIT_GROUP_DEMO)
        set_target_properties(${DEMO_TARGET_NAME} PROPERTIES FOLDER ${IFRIT_GROUP_DEMO})
    endif()
    
    # Set console app property for Windows
    if(DEMO_CONSOLE_APP AND WIN32)
        set_target_properties(${DEMO_TARGET_NAME} PROPERTIES 
            LINK_FLAGS "/SUBSYSTEM:CONSOLE"
        )
    endif()

    # Prebuild step to generate reflection data
    # set(IFRIT_REFLECTION_PARSER_BIN "${CMAKE_SOURCE_DIR}/bin/ifrit.reflparser.exe")
    # message(STATUS "[Ifrit.Demo] Using reflection parser binary: ${IFRIT_REFLECTION_PARSER_BIN}")
    # if(EXISTS ${IFRIT_REFLECTION_PARSER_BIN})
    #     # add_custom_command(TARGET ${DEMO_TARGET_NAME}  PRE_BUILD
    #     #     COMMAND ${IFRIT_REFLECTION_PARSER_BIN} --input "${CMAKE_CURRENT_SOURCE_DIR}/include" --output "${CMAKE_CURRENT_SOURCE_DIR}/include.generated/${DEMO_NAME}.generated.h"
    #     #     COMMENT "Generating reflection data for ${DEMO_NAME}"
    #     # )
    #     set(GENERATE_TARGET_NAME "${DEMO_TARGET_NAME}.reflparse")
    #     add_custom_target(${GENERATE_TARGET_NAME} ALL
    #         COMMAND ${IFRIT_REFLECTION_PARSER_BIN}
    #             --input "${CMAKE_CURRENT_SOURCE_DIR}/include" 
    #             --output "${CMAKE_CURRENT_SOURCE_DIR}/include.generated/${DEMO_NAME}.generated"
    #         COMMENT "Generating core reflection code"
    #         VERBATIM
    #     )
    #     set_target_properties(${GENERATE_TARGET_NAME} PROPERTIES FOLDER ${IFRIT_GROUP_DEMO_GEN})
    #     add_dependencies(${GENERATE_TARGET_NAME} ifrit.reflparser)
    #     add_dependencies(${DEMO_TARGET_NAME} ${GENERATE_TARGET_NAME})
    # else()
    #     message(WARNING "Reflection parser binary not found, skipping reflection data generation for ${DEMO_NAME}")
    # endif()

    
    message(STATUS "[Ifrit.Demo] Created demo target: ${DEMO_TARGET_NAME}")
endfunction()
