set_target_properties(slang PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slangc PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slang-common-objects PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slangd PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slang-glslang PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slangi PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slang-rt PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(core PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(compiler-core PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)

set(IFRIT_SLANG_GENERATED_TARGETS
    slang-capability-defs
    slang-capability-lookup
    slang-fiddle-output
    slang-lookup-tables
    slang-embedded-core-module
    slang-embedded-core-module-source
    slang-no-embedded-core-module
    slang-no-embedded-core-module-source
    slang-glsl-module
    prelude
    copy_slang_headers
    generate_core_module_headers
    copy-slang-llvm
    copy-slang-tint
    copy-webgpu_dawn
)
foreach(target IN LISTS IFRIT_SLANG_GENERATED_TARGETS)
    set_target_properties(${target} PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang/generated)
endforeach()

set(IFRIT_SLANG_GENERATOR_HEADERS
    slang-bootstrap
    slang-lookup-generator
    slang-cpp-parser
    slang-embed
    slang-generate
    slang-spirv-embed-generator
    slang-without-embedded-core-module
    slang-embed
    slang-fiddle
    slang-capability-generator
    all-generators
)
foreach(header IN LISTS IFRIT_SLANG_GENERATOR_HEADERS)
    set_target_properties(${header} PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang/generator)
endforeach()

# external dependencies
set(IFRIT_SLANG_EXTERNAL_DEPENDENCIES
    copy-prebuilt-binaries
    core_tables
    extinst_tables
    lz4_static
    miniz
)
foreach(dep IN LISTS IFRIT_SLANG_EXTERNAL_DEPENDENCIES)
    set_target_properties(${dep} PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang/external)
endforeach()

# external dependencies/glslang
set(IFRIT_SLANG_EXTERNAL_GLSLANG_DEPENDENCIES
    glslang
    SPIRV
)
foreach(dep IN LISTS IFRIT_SLANG_EXTERNAL_GLSLANG_DEPENDENCIES)
    set_target_properties(${dep} PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang/glslang)
endforeach()

# external dependencies/spirv-tools build
set(IFRIT_SLANG_EXTERNAL_SPIRV_TOOLS_DEPENDENCIES
    spirv-tools-build-version
    spirv-tools-header-DebugInfo
    spirv-tools-header-NonSemanticShaderDebugInfo100
    spirv-tools-header-OpenCLDebugInfo100
)
foreach(dep IN LISTS IFRIT_SLANG_EXTERNAL_SPIRV_TOOLS_DEPENDENCIES)
    set_target_properties(${dep} PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang/spirv-tools-build)
endforeach()

# external dependencies/spirv-tools libraries
set(IFRIT_SLANG_EXTERNAL_SPIRV_TOOLS_LIBRARIES
    SPIRV-Tools-link
    SPIRV-Tools-opt
    SPIRV-Tools-static
)
foreach(lib IN LISTS IFRIT_SLANG_EXTERNAL_SPIRV_TOOLS_LIBRARIES)
    set_target_properties(${lib} PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang/spirv-tools-libraries)
endforeach()