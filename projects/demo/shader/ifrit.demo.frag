#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTex;
layout(set = 0, binding = 2) uniform sampler2D texSampler[];

layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform BindlessMapping {
  uint uniformId;
  uint texId;
  uint dummy2;
  uint dummy3;
}
bindlessMapping;

void main() {
  outColor = texture(texSampler[bindlessMapping.texId], fragTex) * 0.9 +
             vec4(fragColor, 1.0) * 0.1;
}