#include "../lib/view_matrix.h"

#include <math.h>

void calculateViewMatrix(float *view,
                         float angleY,
                         float angleX,
                         float distance,
                         float targetX,
                         float targetY,
                         float targetZ) {
    float eyeX = targetX + distance * sinf(angleY) * cosf(angleX);
    float eyeY = targetY + distance * sinf(angleX);
    float eyeZ = targetZ + distance * cosf(angleY) * cosf(angleX);

    float forward[3] = {targetX - eyeX, targetY - eyeY, targetZ - eyeZ};
    float forwardLength =
        sqrtf(forward[0] * forward[0] + forward[1] * forward[1] +
              forward[2] * forward[2]);
    if (forwardLength < 1e-8f)
        forwardLength = 1e-8f;
    forward[0] /= forwardLength;
    forward[1] /= forwardLength;
    forward[2] /= forwardLength;

    float up[3] = {0.0f, 1.0f, 0.0f};

    float side[3] = {forward[1] * up[2] - forward[2] * up[1],
                     forward[2] * up[0] - forward[0] * up[2],
                     forward[0] * up[1] - forward[1] * up[0]};
    float sideLength =
        sqrtf(side[0] * side[0] + side[1] * side[1] + side[2] * side[2]);
    if (sideLength < 1e-8f)
        sideLength = 1e-8f;
    side[0] /= sideLength;
    side[1] /= sideLength;
    side[2] /= sideLength;

    up[0] = side[1] * forward[2] - side[2] * forward[1];
    up[1] = side[2] * forward[0] - side[0] * forward[2];
    up[2] = side[0] * forward[1] - side[1] * forward[0];

    view[0] = side[0];
    view[1] = up[0];
    view[2] = -forward[0];
    view[3] = 0.0f;

    view[4] = side[1];
    view[5] = up[1];
    view[6] = -forward[1];
    view[7] = 0.0f;

    view[8] = side[2];
    view[9] = up[2];
    view[10] = -forward[2];
    view[11] = 0.0f;

    view[12] = -(side[0] * eyeX + side[1] * eyeY + side[2] * eyeZ);
    view[13] = -(up[0] * eyeX + up[1] * eyeY + up[2] * eyeZ);
    view[14] = forward[0] * eyeX + forward[1] * eyeY + forward[2] * eyeZ;
    view[15] = 1.0f;
}
