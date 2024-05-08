#include "presentation/backend/OpenGLBackend.h"

namespace Ifrit::Presentation::Backend {
	void OpenGLBackend::draw() {
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glBindVertexArray(0);
	}
	OpenGLBackend::OpenGLBackend() {
		std::fstream vertexShaderFile(IFRIT_SHADER_PATH"/opengl.backend.vert.glsl", std::ios::in);
		std::fstream fragmentShaderFile(IFRIT_SHADER_PATH"/opengl.backend.frag.glsl", std::ios::in);
		std::string line;

		while (std::getline(vertexShaderFile, line)) {
			vertexShaderCode += line + "\n";
		}
		while (std::getline(fragmentShaderFile, line)) {
			fragmentShaderCode += line + "\n";
		}
		vertexShaderFile.close();
		fragmentShaderFile.close();
		vertexShader = glCreateShader(GL_VERTEX_SHADER);
		const char* vertexShaderCodeCStr = vertexShaderCode.c_str();
		glShaderSource(vertexShader, 1, &vertexShaderCodeCStr, NULL);
		glCompileShader(vertexShader);

		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		const char* fragmentShaderCodeCStr = fragmentShaderCode.c_str();
		glShaderSource(fragmentShader, 1, &fragmentShaderCodeCStr, NULL);
		glCompileShader(fragmentShader);

		// Check for shader compile errors
		int success;
		char infoLog[512];
		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
			prints("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" , infoLog);
		}
		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
			prints("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" , infoLog);
		}


		shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);
		glUseProgram(shaderProgram);

		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

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
	void OpenGLBackend::updateTexture(Ifrit::Core::Data::ImageU8 image) {
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.getWidth(), image.getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image.getData());
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	void OpenGLBackend::setViewport(int32_t x, int32_t y, int32_t width, int32_t height) {
		glViewport(x, y, width, height);
	}
}