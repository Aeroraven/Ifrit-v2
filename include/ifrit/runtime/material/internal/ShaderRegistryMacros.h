#pragma once
#define REG_SHADER(name, path, stage) shaderRegistry->RegisterShader(name, path, "main", stage)
#define REG_COMPUTE(name, path) REG_SHADER(name, path ".comp.glsl", RHI::RhiShaderStage::Compute)
#define REG_VERTEX(name, path) REG_SHADER(name, path ".vert.glsl", RHI::RhiShaderStage::Vertex)
#define REG_FRAGMENT(name, path) REG_SHADER(name, path ".frag.glsl", RHI::RhiShaderStage::Fragment)
#define REG_MESH(name, path) REG_SHADER(name, path ".mesh.glsl", RHI::RhiShaderStage::Mesh)

#define REG_SHADER_NEO(name, path, stage, entry) shaderRegistry->RegisterShader(name, path, entry, stage)
#define REG_COMPUTE_NEO(name, path, entry) REG_SHADER_NEO(name, path ".comp.slang", RHI::RhiShaderStage::Compute, entry)
#define REG_VERTEX_NEO(name, path, entry) REG_SHADER_NEO(name, path ".vert.slang", RHI::RhiShaderStage::Vertex, entry)
#define REG_FRAGMENT_NEO(name, path, entry) \
    REG_SHADER_NEO(name, path ".frag.slang", RHI::RhiShaderStage::Fragment, entry)
#define REG_MESH_NEO(name, path, entry) REG_SHADER_NEO(name, path ".mesh.slang", RHI::RhiShaderStage::Mesh, entry)