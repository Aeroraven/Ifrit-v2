# PostBuild: Convert MinGW libraries to MSVC .lib files
macro(WindowsPostbuild target defFile dllFile libFile outputDir arch)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    
    if (MINGW)
        message(STATUS "Renaming ${dllFile}.a to ${TEMP_LIBFILE_NOSUFFIX}.a")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND dlltool --input-def ${defFile} --dllname ${dllFile} --output-lib ${libFile}
            WORKING_DIRECTORY ${outputDir}
        )
        add_custom_command(
            TARGET ${target} POST_BUILD
            COMMAND lib /def:${defFile} /out:${libFile} /machine:${arch}
            WORKING_DIRECTORY ${outputDir}
        )
        message(STATUS "[IFRIT/Win]: Postbuild: ${defFile} -> ${libFile}")
    endif()
endif()
endif()
endmacro()

# PreBuild: Delete built files with given prefix
macro(WindowsPrebuild target outputDir prefix)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if (MINGW)
        add_custom_command(TARGET ${target} PRE_BUILD
            COMMAND cmd /c del ${prefix}.*
            WORKING_DIRECTORY ${outputDir}
        )
    endif()
endif()
endif()
message(STATUS "[IFRIT/Win]: Prebuild: ${prefix}*")
endmacro()

# AddImpLib: Add import library for MinGW
macro(AddImpLib target outputDir libName)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if (MINGW)
        set_target_properties(${target} PROPERTIES
            LINK_FLAGS "-Wl,--subsystem,windows,--output-def,${outputDir}/${libName}.def,--out-implib,${libName}.a"
        )
        message(STATUS "[IFRIT/Win]: Add import library: ${libName}.a")
    endif()
endif()
endif()
endmacro()