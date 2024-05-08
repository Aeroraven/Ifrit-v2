#pragma once
#include "dependency/GLAD/glad/glad.h"
#include "presentation/backend/BackendProvider.h"
#include "core/definition/CoreExports.h"
#include "core/data/Image.h"

namespace Ifrit::Presentation::Backend {
	class OpenGLBackend : public BackendProvider {
	private:
		std::string vertexShaderCode;
		std::string fragmentShaderCode;
		GLuint vertexShader;
		GLuint fragmentShader;
		GLuint shaderProgram;

		std::vector<float> vertices = {
			-1.0f, -1.0f, 0.0f,
			-1.0, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,
			-1.0f, -1.0f, 0.0f,
		};

		std::vector<float> texCoords = {
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,
		};

		GLuint VAO = 0;
		GLuint VBO = 0;
		GLuint texture = 0;

	public:
		OpenGLBackend();
		void draw();
		void updateTexture(Ifrit::Core::Data::ImageU8 image);
		void setViewport(int32_t x, int32_t y, int32_t width, int32_t height);
	};
}