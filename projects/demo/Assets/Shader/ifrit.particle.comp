#version 450
#extension GL_EXT_nonuniform_qualifier : enable

// modified from  https://vulkan-tutorial.com/code/31_shader_compute.comp

struct Particle {
  vec2 position;
  vec2 velocity;
  vec4 color;
};

layout(std140, set = 0, binding = 1) buffer ParticleSSBOIn { Particle p[1024]; }
particles[];

layout(set = 1, binding = 0) uniform BindlessMapping {
  uint ssboIn;
  uint ssboOut;
  uint dummy1;
  uint dummy2;
}
bindlessMapping;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void main() {
  uint index = gl_GlobalInvocationID.x;

  Particle particleIn = particles[bindlessMapping.ssboIn].p[index];
  Particle particleOut;

  particleOut.position = particleIn.position + particleIn.velocity.xy * 0.01;
  particleOut.velocity = particleIn.velocity;

  // Flip movement at window border
  if ((particleOut.position.x <= -1.0) || (particleOut.position.x >= 1.0)) {
    particleOut.velocity.x = -particleOut.velocity.x;
  }
  if ((particleOut.position.y <= -1.0) || (particleOut.position.y >= 1.0)) {
    particleOut.velocity.y = -particleOut.velocity.y;
  }

  particles[bindlessMapping.ssboOut].p[index].position = particleOut.position;
  particles[bindlessMapping.ssboOut].p[index].velocity = particleOut.velocity;
  particles[bindlessMapping.ssboOut].p[index].color = particleIn.color;
}