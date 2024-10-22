#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0) uniform UniformBufferObject { mat4 mvp; }
ubo[];

layout(set = 1, binding = 0) uniform BindlessMapping {
  uint uniformId;
  uint dummy1;
  uint dummy2;
  uint dummy3;
}
bindlessMapping;

layout(location = 0) out vec3 fragColor;

void main() {
  vec4 pos = vec4(inPosition, 0.0, 1.0);
  gl_Position = ubo[bindlessMapping.uniformId].mvp * pos;
  fragColor = inColor;
}