#version 430 core

// Trail buffer is bound as a VBO: each vertex is vec4(x, y, z, speed).
layout(location = 0) in vec4 posSpeed;

uniform mat4 projection;
uniform mat4 view;
uniform int trailLen;

out float vSpeed;
out float vAlpha;

void main() {
    gl_Position = projection * view * vec4(posSpeed.xyz, 1.0);
    vSpeed = posSpeed.w;

    // Fade along the trail based on position within the strip.
    // gl_VertexID % trailLen gives the age: 0 = newest, trailLen-1 = oldest.
    int age = gl_VertexID % trailLen;
    vAlpha = 1.0 - float(age) / float(trailLen);
}
