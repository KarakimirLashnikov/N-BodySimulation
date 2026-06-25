#version 450
// Particle point rendering – fragment shader

layout(location = 0) out vec4 outColor;

void main() {
    // Simple warm orange-yellow glow
    outColor = vec4(1.0, 0.75, 0.2, 1.0);
}
