#version 330
layout(location = 0)in vec4 position;
layout(location = 1)in vec4 normal;
layout(std140) uniform matVP  
{  
   mat4 mvp;
};  

out vec4 outnormal;

void main()
{
	gl_Position = mvp * position;
	outnormal = normal;
}