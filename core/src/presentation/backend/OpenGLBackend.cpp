#include "presentation/backend/OpenGLBackend.h"

namespace Ifrit::Presentation::Backend {
	void OpenGLBackend::draw() {
		glDepthFunc(GL_ALWAYS);
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glDrawArrays(GL_TRIANGLES, 0, 6);
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
			printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED %s\n" , infoLog);
		}
		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
			printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED %s\n" , infoLog);
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
	void OpenGLBackend::updateTexture(const Ifrit::Core::Data::ImageF32& image) {
		const static float* ptr = nullptr;
		glBindTexture(GL_TEXTURE_2D, texture);
		auto data = image.getData();
		if (ptr != data) IFRIT_BRANCH_UNLIKELY {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.getWidth(), image.getHeight(), 0, GL_RGBA, GL_FLOAT, image.getData());
			ptr = image.getData();
		}
		else {
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.getWidth(), image.getHeight(), GL_RGBA, GL_FLOAT, data);
		}
		//glGenerateMipmap(GL_TEXTURE_2D);
	}
	void OpenGLBackend::setViewport(int32_t x, int32_t y, int32_t width, int32_t height) {
		glViewport(x, y, width, height);
	}
}