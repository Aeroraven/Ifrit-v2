#version 330
layout(location=0)in vec3 position;

out vec2 texc;

void main()
{
    gl_Position =vec4(position, 1.0);
    texc = position.xy*0.5+0.5;
}