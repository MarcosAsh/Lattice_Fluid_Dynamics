#version 430 core

in float vSpeed;
in float vLife;
in vec3 vVelocity;
in vec3 vPosition;

out vec4 FragColor;

uniform int visualizationMode; // 0=depth, 1=velocity mag, 2=velocity dir,
                               // 3=life, 4=pressure zones
uniform float maxSpeed;        // For normalizing velocity colors

// HSV to RGB conversion for smooth color mapping
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Jet colormap (blue -> cyan -> green -> yellow -> red)
vec3 jetColormap(float t) {
    t = clamp(t, 0.0, 1.0);

    vec3 color;
    if (t < 0.25) {
        float s = t / 0.25;
        color = vec3(0.0, s, 1.0); // blue to cyan
    } else if (t < 0.5) {
        float s = (t - 0.25) / 0.25;
        color = vec3(0.0, 1.0, 1.0 - s); // cyan to green
    } else if (t < 0.75) {
        float s = (t - 0.5) / 0.25;
        color = vec3(s, 1.0, 0.0); // green to yellow
    } else {
        float s = (t - 0.75) / 0.25;
        color = vec3(1.0, 1.0 - s, 0.0); // yellow to red
    }
    return color;
}

// Cool-warm diverging colormap
vec3 coolWarmColormap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 cool = vec3(0.2, 0.4, 0.8); // blue
    vec3 warm = vec3(0.8, 0.2, 0.2); // red
    vec3 mid = vec3(0.9, 0.9, 0.9);  // white/gray

    if (t < 0.5) {
        return mix(cool, mid, t * 2.0);
    } else {
        return mix(mid, warm, (t - 0.5) * 2.0);
    }
}

// Viridis-like colormap
vec3 viridisColormap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.004, 0.329);
    vec3 c1 = vec3(0.282, 0.140, 0.458);
    vec3 c2 = vec3(0.254, 0.265, 0.530);
    vec3 c3 = vec3(0.207, 0.372, 0.553);
    vec3 c4 = vec3(0.164, 0.471, 0.558);
    vec3 c5 = vec3(0.128, 0.567, 0.551);
    vec3 c6 = vec3(0.135, 0.659, 0.518);
    vec3 c7 = vec3(0.267, 0.749, 0.441);
    vec3 c8 = vec3(0.478, 0.821, 0.318);
    vec3 c9 = vec3(0.741, 0.873, 0.150);
    vec3 c10 = vec3(0.993, 0.906, 0.144);

    float idx = t * 10.0;
    int i = int(floor(idx));
    float f = fract(idx);

    vec3 colors[11] = vec3[](c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10);

    if (i >= 10)
        return c10;
    return mix(colors[i], colors[i + 1], f);
}

void main() {
    vec3 color;
    float alpha = 1.0;

    // Mode 0: Depth-based coloring (original)
    if (visualizationMode == 0) {
        float depth = gl_FragCoord.z;
        color = mix(vec3(0.5, 0.7, 1.0), vec3(1.0), 1.0 - depth);
    }
    // Mode 1: Velocity magnitude (jet colormap)
    else if (visualizationMode == 1) {
        float normalizedSpeed = clamp(vSpeed / maxSpeed, 0.0, 1.0);
        color = jetColormap(normalizedSpeed);
    }
    // Mode 2: Velocity direction (RGB = normalized velocity XYZ)
    else if (visualizationMode == 2) {
        vec3 normVel = normalize(vVelocity);
        color = (normVel + 1.0) * 0.5; // Map [-1,1] to [0,1]
    }
    // Mode 3: Particle lifetime (viridis colormap)
    else if (visualizationMode == 3) {
        color = viridisColormap(vLife);
        alpha = 0.5 + 0.5 * vLife; // Fade as particles age
    }
    // Mode 4: Pressure zones / turbulence indicator
    else if (visualizationMode == 4) {
        // Use position and velocity to estimate turbulence
        float turbulence =
            length(vec2(vVelocity.y, vVelocity.z)) / (abs(vVelocity.x) + 0.01);
        turbulence = clamp(turbulence * 2.0, 0.0, 1.0);
        color = coolWarmColormap(turbulence);
    }
    // Mode 5: Streamline coloring (based on x-position for flow progress)
    else if (visualizationMode == 5) {
        float progress =
            (vPosition.x + 4.0) / 8.0; // Normalize x from [-4, 4] to [0, 1]
        color = hsv2rgb(
            vec3(progress * 0.7, 0.8, 0.9)); // Rainbow based on progress
    }
    // Mode 6: Vorticity indicator (lateral motion)
    else if (visualizationMode == 6) {
        float lateral =
            sqrt(vVelocity.y * vVelocity.y + vVelocity.z * vVelocity.z);
        float axial = abs(vVelocity.x) + 0.001;
        float vorticity = lateral / axial;
        vorticity = clamp(vorticity * 3.0, 0.0, 1.0);

        // Purple for laminar, orange for turbulent
        vec3 laminar = vec3(0.4, 0.2, 0.8);
        vec3 turbulent = vec3(1.0, 0.5, 0.0);
        color = mix(laminar, turbulent, vorticity);
    }
    // Mode 8: Surface pressure distribution (Cp via cool-warm)
    else if (visualizationMode == 8) {
        // vSpeed holds Cp (pressure coefficient), typically in [-2, 2].
        // Map: -1 = blue (low pressure), 0 = white, +1 = red (high pressure).
        float cp = clamp(vSpeed * 0.5 + 0.5, 0.0, 1.0);
        color = coolWarmColormap(cp);
    }
    // Default: white
    else {
        color = vec3(1.0);
    }

    FragColor = vec4(color, alpha);
}