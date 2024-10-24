#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(set = 0, binding = 0) uniform UniformBufferObject { mat4 mvp; }
ubo[];

layout(set = 1, binding = 0) uniform BindlessMapping {
  uint uniformId;
  uint texId;
  uint dummy2;
  uint dummy3;
}
bindlessMapping;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTex;

void main() {
  vec4 pos = vec4(inPosition, 1.0);
  gl_Position = ubo[bindlessMapping.uniformId].mvp * pos;
  fragColor = inColor;
  fragTex = inTexCoord;
}