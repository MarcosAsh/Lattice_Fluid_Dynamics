#ifndef VIEW_MATRIX_H
#define VIEW_MATRIX_H

// Build a column-major 4x4 view matrix looking from an orbit
// position (angleY, angleX, distance) at (targetX, targetY, targetZ).
void calculateViewMatrix(float *view,
                         float angleY,
                         float angleX,
                         float distance,
                         float targetX,
                         float targetY,
                         float targetZ);

#endif // VIEW_MATRIX_H
