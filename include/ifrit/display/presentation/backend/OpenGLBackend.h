
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


#pragma once
#include "ifrit/display/dependencies/GLAD/glad/glad.h"
#include "ifrit/display/presentation/backend/BackendProvider.h"
#include <cstdint>
#include <string>
#include <vector>

namespace Ifrit::Display::Backend {
class IFRIT_APIDECL OpenGLBackend : public BackendProvider {
private:
  std::string vertexShaderCode;
  std::string fragmentShaderCode;
  GLuint vertexShader;
  GLuint fragmentShader;
  GLuint shaderProgram;

  std::vector<float> vertices = {
      -1.0f, -1.0f, 0.0f, -1.0, 1.0f,  0.0f, 1.0f,  1.0f,  0.0f,
      1.0f,  1.0f,  0.0f, 1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 0.0f,
  };

  std::vector<float> texCoords = {
      0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  };

  GLuint VAO = 0;
  GLuint VBO = 0;
  GLuint texture = 0;

public:
  OpenGLBackend();
  virtual void draw() override;
  virtual void updateTexture(const float *image, int channels, int width,
                             int height) override;
  virtual void setViewport(int32_t x, int32_t y, int32_t width,
                           int32_t height) override;
};
} // namespace Ifrit::Display::Backend