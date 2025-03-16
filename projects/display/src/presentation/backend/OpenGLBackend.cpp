
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "ifrit/display/presentation/backend/OpenGLBackend.h"

static const char *vertexShaderCodeF = R"(
#version 330
layout(location=0)in vec3 position;

out vec2 texc;

void main()
{
    gl_Position =vec4(position, 1.0);
    texc = position.xy*0.5+0.5;
}
)";
static const char *fragmentShaderCodeF = R"(
#version 330
precision lowp float;
in vec2 texc;
out vec4 fragColor;

uniform sampler2D tex;
void main() {
    fragColor = texture(tex, texc);
}
)";

namespace Ifrit::Display::Backend {
IFRIT_APIDECL void OpenGLBackend::draw() {
  glDepthFunc(GL_ALWAYS);
  glUseProgram(shaderProgram);
  glBindVertexArray(VAO);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glDrawArrays(GL_TRIANGLES, 0, 6);
}
IFRIT_APIDECL OpenGLBackend::OpenGLBackend() {
  vertexShaderCode = vertexShaderCode.empty() ? vertexShaderCodeF : vertexShaderCode;
  fragmentShaderCode = fragmentShaderCode.empty() ? fragmentShaderCodeF : fragmentShaderCode;
  vertexShader = glCreateShader(GL_VERTEX_SHADER);
  const char *vertexShaderCodeCStr = vertexShaderCode.c_str();
  glShaderSource(vertexShader, 1, &vertexShaderCodeCStr, NULL);
  glCompileShader(vertexShader);

  fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  const char *fragmentShaderCodeCStr = fragmentShaderCode.c_str();
  glShaderSource(fragmentShader, 1, &fragmentShaderCodeCStr, NULL);
  glCompileShader(fragmentShader);

  // Check for shader compile errors
  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED %s\n", infoLog);
  }
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED %s\n", infoLog);
  }

  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  glUseProgram(shaderProgram);

  int linkSuccess;
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linkSuccess);
  if (!linkSuccess) {
    glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    printf("ERROR::SHADER::PROGRAM::LINK_FAILED %s\n", infoLog);
  }

  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}
IFRIT_APIDECL void OpenGLBackend::updateTexture(const float *image, int channels, int width, int height) {
  const static float *ptr = nullptr;
  glBindTexture(GL_TEXTURE_2D, texture);
  auto data = image;
  if (ptr != data) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, data);
    ptr = image;
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, data);
  }
  // glGenerateMipmap(GL_TEXTURE_2D);
}
IFRIT_APIDECL void OpenGLBackend::setViewport(int32_t x, int32_t y, int32_t width, int32_t height) {
  glViewport(x, y, width, height);
}
} // namespace Ifrit::Display::Backend