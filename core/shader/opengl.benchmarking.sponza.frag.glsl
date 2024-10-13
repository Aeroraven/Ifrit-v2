#version 330
precision lowp float;

in vec4 outnormal;
out vec4 fragColor;

void main() {
	fragColor = vec4(outnormal.xyz, 1.0)*0.5+0.5;
}