#define _USE_MATH_DEFINES
#include "../lib/render_model.h"
#include "../lib/config.h"
#include <glad/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void renderModel(Model *model) {
    if (model->faceCount == 0)
        return;

    glPushMatrix();

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3f(1.0f, 1.0f, 0.0f);

    float scaleF = g_modelScale;
    float offsetX = g_offsetX;
    float offsetY = g_offsetY;
    float offsetZ = g_offsetZ;

    float radY = g_carRotationY * (float)M_PI / 180.0f;
    float cosY = cosf(radY);
    float sinY = sinf(radY);

    glBegin(GL_TRIANGLES);
    for (int i = 0; i < model->faceCount; i++) {
        int idx0 = model->faces[i].v1 - 1;
        int idx1 = model->faces[i].v2 - 1;
        int idx2 = model->faces[i].v3 - 1;

        if (idx0 < 0 || idx0 >= model->vertexCount || idx1 < 0 ||
            idx1 >= model->vertexCount || idx2 < 0 ||
            idx2 >= model->vertexCount) {
            continue;
        }

        Vertex v0 = model->vertices[idx0];
        Vertex v1 = model->vertices[idx1];
        Vertex v2 = model->vertices[idx2];

        float x0 = v0.x * scaleF + offsetX;
        float y0 = v0.y * scaleF + offsetY;
        float z0 = v0.z * scaleF + offsetZ;

        float x1 = v1.x * scaleF + offsetX;
        float y1 = v1.y * scaleF + offsetY;
        float z1 = v1.z * scaleF + offsetZ;

        float x2 = v2.x * scaleF + offsetX;
        float y2 = v2.y * scaleF + offsetY;
        float z2 = v2.z * scaleF + offsetZ;

        float rx0 = x0 * cosY - z0 * sinY;
        float rz0 = x0 * sinY + z0 * cosY;

        float rx1 = x1 * cosY - z1 * sinY;
        float rz1 = x1 * sinY + z1 * cosY;

        float rx2 = x2 * cosY - z2 * sinY;
        float rz2 = x2 * sinY + z2 * cosY;

        glVertex3f(rx0, y0, rz0);
        glVertex3f(rx1, y1, rz1);
        glVertex3f(rx2, y2, rz2);
    }
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glPopMatrix();
}