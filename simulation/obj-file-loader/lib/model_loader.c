#include "../lib/model_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Largest polygon we fan-triangulate; ample for any real OBJ face.
#define MAX_FACE_VERTS 64

static int countVerticesInFile(const char *filePath) {
    FILE *file = fopen(filePath, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filePath);
        return 0;
    }

    int vertexCount = 0;
    // Same buffer size as loadOBJ so an over-long line splits identically
    // in both passes and the counts cannot disagree.
    char line[1024];

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            vertexCount++;
        }
    }
    fclose(file);
    return vertexCount;
}

// Number of whitespace-separated vertex tokens on a face line (after 'f').
static int countFaceVerts(const char *line) {
    const char *p = line + 1; // skip the leading 'f'
    int n = 0;
    while (*p) {
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '\0' || *p == '\n' || *p == '\r')
            break;
        n++;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
            p++;
        }
    }
    return n;
}

// Triangle count after fan-triangulating every face (k verts -> k-2 tris).
static int countTrianglesInFile(const char *filePath) {
    FILE *file = fopen(filePath, "r");
    if (!file) {
        printf("Error: could not open file %s\n", filePath);
        return 0;
    }

    int triangleCount = 0;
    char line[1024];

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'f' && line[1] == ' ') {
            int nv = countFaceVerts(line);
            if (nv >= 3) {
                triangleCount += nv - 2;
            }
        }
    }

    fclose(file);
    return triangleCount;
}

// Parse a single vertex index from OBJ face format
// Handles: "v", "v/vt", "v/vt/vn", "v//vn"
static int parseVertexIndex(const char *token) {
    int v = 0;
    sscanf(token, "%d", &v); // Just get the first number
    return v;
}

Model loadOBJ(const char *filePath) {
    Model model = {0};

    int vertexCount = countVerticesInFile(filePath);
    int triangleCount = countTrianglesInFile(filePath);

    printf("File contains %d vertices and %d triangles.\n",
           vertexCount,
           triangleCount);

    model.vertices = (Vertex *)malloc(vertexCount * sizeof(Vertex));
    model.faces = (Face *)malloc(triangleCount * sizeof(Face));

    if ((vertexCount > 0 && !model.vertices) ||
        (triangleCount > 0 && !model.faces)) {
        printf("Error: Out of memory!\n");
        free(model.vertices);
        free(model.faces);
        return (Model){0};
    }

    FILE *file = fopen(filePath, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filePath);
        free(model.vertices);
        free(model.faces);
        return (Model){0};
    }

    char line[1024];
    int vertexIndex = 0, faceIndex = 0;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            // Parse vertex (guarded like the face write below).
            if (vertexIndex < vertexCount) {
                sscanf(line,
                       "v %f %f %f",
                       &model.vertices[vertexIndex].x,
                       &model.vertices[vertexIndex].y,
                       &model.vertices[vertexIndex].z);
                vertexIndex++;
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            // Parse face. Handles f v, f v/vt, f v/vt/vn, and f v//vn, with
            // any number of vertices. Polygons are fan-triangulated
            // (v0, vk, vk+1); the previous loader kept only each face's
            // first triangle, silently dropping half of every quad.
            int idx[MAX_FACE_VERTS];
            int nv = 0;

            char *ptr = line + 1; // skip 'f'
            while (*ptr && nv < MAX_FACE_VERTS) {
                while (*ptr == ' ' || *ptr == '\t')
                    ptr++;
                if (*ptr == '\0' || *ptr == '\n' || *ptr == '\r')
                    break;

                char tok[64];
                int i = 0;
                while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' &&
                       *ptr != '\r' && i < 63) {
                    tok[i++] = *ptr++;
                }
                tok[i] = '\0';
                idx[nv++] = parseVertexIndex(tok);
            }

            for (int k = 1; k + 1 < nv && faceIndex < triangleCount; k++) {
                model.faces[faceIndex].v1 = idx[0];
                model.faces[faceIndex].v2 = idx[k];
                model.faces[faceIndex].v3 = idx[k + 1];
                faceIndex++;
            }
        }
    }

    fclose(file);

    model.vertexCount = vertexIndex;
    model.faceCount = faceIndex;

    printf("Successfully loaded model with %d vertices and %d triangles.\n",
           vertexIndex,
           faceIndex);

    return model;
}

void freeModel(Model *model) {
    free(model->vertices);
    free(model->faces);
    model->vertices = NULL;
    model->faces = NULL;
    model->vertexCount = 0;
    model->faceCount = 0;
}
