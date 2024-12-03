
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


#include "OglBenchmarking.h"
#include "dependency/GLAD/glad/glad.h"
#include "ifrit/common/math/LinalgOps.h"
#include "presentation/window/GLFWWindowProvider.h"
#include "utility/loader/WavefrontLoader.h"
using namespace Ifrit::SoftRenderer::Utility::Loader;
using namespace Ifrit::Display::Window;
using namespace Ifrit::Math;
namespace Ifrit::Demo::OglBenchmarking {
int mainCpu() {
  // load sponza
  WavefrontLoader loader;
  std::vector<ifloat3> pos;
  std::vector<ifloat3> normal;
  std::vector<ifloat2> uv;
  std::vector<uint32_t> index;
  std::vector<ifloat3> procNormal;
  loader.loadObject(IFRIT_ASSET_PATH "/bunny.obj", pos, normal, uv, index);
  procNormal = loader.remapNormals(normal, index, pos.size());

  // load shader
  std::string vertexShaderCode;
  std::string fragmentShaderCode;
  std::fstream vertexShaderFile(
      IFRIT_SHADER_PATH "/opengl.benchmarking.sponza.vert.glsl", std::ios::in);
  std::fstream fragmentShaderFile(
      IFRIT_SHADER_PATH "/opengl.benchmarking.sponza.frag.glsl", std::ios::in);
  std::string line;

  while (std::getline(vertexShaderFile, line)) {
    vertexShaderCode += line + "\n";
  }
  while (std::getline(fragmentShaderFile, line)) {
    fragmentShaderCode += line + "\n";
  }
  vertexShaderFile.close();
  fragmentShaderFile.close();

  // create indexbuffer (preprocess)
  std::vector<int> indexBuffer;
  indexBuffer.resize(index.size() / 3);
  for (int i = 0; i < index.size(); i += 3) {
    indexBuffer[i / 3] = index[i];
  }

  std::vector<ifloat4> pos4;
  pos4.resize(pos.size());
  for (int i = 0; i < pos.size(); i++) {
    pos4[i] = ifloat4(pos[i].x, pos[i].y, pos[i].z, 1);
  }

  std::vector<ifloat4> normal4;
  normal4.resize(procNormal.size());
  for (int i = 0; i < procNormal.size(); i++) {
    normal4[i] =
        ifloat4(procNormal[i].x, procNormal[i].y, procNormal[i].z, 0.5f);
  }

  GLFWWindowProvider windowProvider;
  windowProvider.setup(2048, 2048);
  windowProvider.setTitle("Ifrit-v2 OpenGL Benchmark");

  // opengl: create vaos, vbos, ebo
  unsigned int VAO, VBO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, pos4.size() * sizeof(ifloat4), pos4.data(),
               GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.size() * sizeof(int),
               indexBuffer.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  // new VBO for normals
  unsigned int VBO2;
  glGenBuffers(1, &VBO2);
  glBindBuffer(GL_ARRAY_BUFFER, VBO2);
  glBufferData(GL_ARRAY_BUFFER, normal4.size() * sizeof(ifloat4),
               normal4.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(1);

  // uniform buffer & bind
  float4x4 view = (lookAt({0, 0.1, 0.25}, {0, 0.1, 0.0}, {0, 1, 0}));
  // float4x4 view = (lookAt({ 0,0.75,1.50 }, { 0,0.75,0.0 }, { 0,1,0 }));
  // float4x4 view = (lookAt({ 500,300,0 }, { -100,300,-0 }, { 0,1,0 }));
  // float4x4 view = (lookAt({ 0,1.5,0 }, { -100,1.5,0 }, { 0,1,0 }));
  float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.1, 3000));
  float4x4 mvp = transpose(matmul(proj, view));

  // opengl: create shaders
  unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
  const char *vertexShaderCodeCStr = vertexShaderCode.c_str();
  glShaderSource(vertexShader, 1, &vertexShaderCodeCStr, NULL);
  glCompileShader(vertexShader);

  unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
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

  unsigned int shaderProgram = glCreateProgram();
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

  GLint nMatVPBlockIndex = glGetUniformBlockIndex(shaderProgram, "matVP");
  printf("%d uniform: %d\n", nMatVPBlockIndex, glGetError());
  unsigned int uniformBuffer;
  glGenBuffers(1, &uniformBuffer);
  glBindBuffer(GL_UNIFORM_BUFFER, uniformBuffer);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(float4x4), &mvp, GL_STATIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniformBuffer);

  // opengl: render loop

  glDepthFunc(GL_LEQUAL);
  glViewport(0, 0, 2048, 2048);
  glEnable(GL_DEPTH_TEST);

  windowProvider.loop([&](int *coreTime) {
    auto start = std::chrono::high_resolution_clock::now();
    glClearColor(0.0f, 0.3f, 0.3f, 1.0f);
    glClearDepth(255.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glBindBuffer(GL_UNIFORM_BUFFER, uniformBuffer);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniformBuffer);
    glDrawElements(GL_TRIANGLES, indexBuffer.size(), GL_UNSIGNED_INT, 0);
    auto end = std::chrono::high_resolution_clock::now();
    *coreTime =
        (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
  });
  return 0;
}
} // namespace Ifrit::Demo::OglBenchmarking
