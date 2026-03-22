#version 430 core

in float vSpeed;
in float vAlpha;

out vec4 FragColor;

uniform float maxSpeed;

vec3 jetColormap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 color;
    if (t < 0.25) {
        float s = t / 0.25;
        color = vec3(0.0, s, 1.0);
    } else if (t < 0.5) {
        float s = (t - 0.25) / 0.25;
        color = vec3(0.0, 1.0, 1.0 - s);
    } else if (t < 0.75) {
        float s = (t - 0.5) / 0.25;
        color = vec3(s, 1.0, 0.0);
    } else {
        float s = (t - 0.75) / 0.25;
        color = vec3(1.0, 1.0 - s, 0.0);
    }
    return color;
}

void main() {
    vec3 color = jetColormap(vSpeed / maxSpeed);
    FragColor = vec4(color, vAlpha * 0.8);
}
