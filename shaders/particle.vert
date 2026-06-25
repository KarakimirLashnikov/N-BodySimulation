#version 450
// Particle point rendering – vertex shader

// ---------------------------------------------------------------
// Particle – must match C++ Particle struct (32 bytes)
// ---------------------------------------------------------------
struct Particle {
    vec4 position;  // xyz = position, w = mass
    vec4 velocity;  // xyz = velocity, w = unused
};

layout(std430, binding = 1) readonly buffer ParticleBuffer {
    Particle particles[];
};

// ---------------------------------------------------------------
// Push constants – MVP matrix
// ---------------------------------------------------------------
layout(push_constant) uniform PushBlock {
    mat4 mvp;
} push;

// ---------------------------------------------------------------
// Main
// ---------------------------------------------------------------
void main() {
    Particle p = particles[gl_VertexIndex];
    gl_Position = push.mvp * vec4(p.position.xyz, 1.0);
    gl_PointSize = 2.5;
}
