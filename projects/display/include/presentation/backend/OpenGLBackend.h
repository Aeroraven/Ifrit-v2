#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "dependencies/GLAD/glad/glad.h"
#include "presentation/backend/BackendProvider.h"

namespace Ifrit::Presentation::Backend {
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
  virtual void updateTexture(const float* image, int channels,int width,int height) override;
  virtual void setViewport(int32_t x, int32_t y, int32_t width,
                           int32_t height) override;
};
} // namespace Ifrit::Presentation::Backend