#version 330
precision lowp float;

in vec2 texc;
out vec4 fragColor;

uniform sampler2D tex;
void main() {
    fragColor = texture(tex, texc);
}