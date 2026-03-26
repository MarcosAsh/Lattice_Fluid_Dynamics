#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in float flowSpeed; // LBM field speed (stored in padding1)
layout(location = 2) in vec3 velocity;
layout(location = 3) in float life;

uniform mat4 projection;
uniform mat4 view;

out float vSpeed;
out float vLife;
out vec3 vVelocity;
out vec3 vPosition;

void main() {
    gl_Position = projection * view * vec4(position, 1.0);

    // Use LBM flow speed for coloring -- smooth, no oscillation
    vSpeed = flowSpeed;
    vLife = life;
    vVelocity = velocity;
    vPosition = position;

    // Size varies with distance for depth perception
    float dist = length(gl_Position.xyz);
    gl_PointSize = max(3.0, 8.0 / (dist * 0.5));
}