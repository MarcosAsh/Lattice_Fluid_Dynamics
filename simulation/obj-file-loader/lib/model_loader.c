#include "../lib/model_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int countVerticesInFile(const char* filePath) {
    FILE* file = fopen(filePath, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filePath);
        return 0;
    }

    int vertexCount = 0;
    char line[256];

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            vertexCount++;
        }
    }
    fclose(file);
    return vertexCount;
}

int countFacesInFile(const char* filePath) {
    FILE* file = fopen(filePath, "r");
    if (!file) {
        printf("Error: could not open file %s\n", filePath);
        return 0;
    }

    int faceCount = 0;
    char line[256];

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'f' && line[1] == ' ') {
            faceCount++;
        }
    }

    fclose(file);
    return faceCount;
}

// Parse a single vertex index from OBJ face format
// Handles: "v", "v/vt", "v/vt/vn", "v//vn"
int parseVertexIndex(const char* token) {
    int v = 0;
    sscanf(token, "%d", &v);  // Just get the first number
    return v;
}

Model loadOBJ(const char* filePath) {
    Model model = {0};

    int vertexCount = countVerticesInFile(filePath);
    int faceCount = countFacesInFile(filePath);

    printf("File contains %d vertices and %d faces.\n", vertexCount, faceCount);

    model.vertices = (Vertex*)malloc(vertexCount * sizeof(Vertex));
    model.faces = (Face*)malloc(faceCount * sizeof(Face));

    if (!model.vertices || !model.faces) {
        printf("Error: Out of memory!\n");
        free(model.vertices);
        free(model.faces);
        return model;
    }

    FILE* file = fopen(filePath, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filePath);
        free(model.vertices);
        free(model.faces);
        return model;
    }

    char line[512];
    int vertexIndex = 0, faceIndex = 0;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            // Parse vertex
            sscanf(line, "v %f %f %f", 
                   &model.vertices[vertexIndex].x,
                   &model.vertices[vertexIndex].y,
                   &model.vertices[vertexIndex].z);
            vertexIndex++;
        } 
        else if (line[0] == 'f' && line[1] == ' ') {
            // Parse face handles multiple formats:
            // f v1 v2 v3
            // f v1/vt1 v2/vt2 v3/vt3
            // f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            // f v1//vn1 v2//vn2 v3//vn3
            
            char v1str[64] = {0}, v2str[64] = {0}, v3str[64] = {0};
            
            // Skip f and parse the three vertex tokens
            char* ptr = line + 2;
            
            // Skip leading whitespace
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            
            // Read first token
            int i = 0;
            while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && i < 63) {
                v1str[i++] = *ptr++;
            }
            v1str[i] = '\0';
            
            // Skip whitespace
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            
            // Read second token
            i = 0;
            while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && i < 63) {
                v2str[i++] = *ptr++;
            }
            v2str[i] = '\0';
            
            // Skip whitespace
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            
            // Read third token
            i = 0;
            while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && i < 63) {
                v3str[i++] = *ptr++;
            }
            v3str[i] = '\0';
            
            // Parse vertex indices (handles v/vt/vn format)
            model.faces[faceIndex].v1 = parseVertexIndex(v1str);
            model.faces[faceIndex].v2 = parseVertexIndex(v2str);
            model.faces[faceIndex].v3 = parseVertexIndex(v3str);
            
            faceIndex++;
        }
    }

    fclose(file);

    model.vertexCount = vertexCount;
    model.faceCount = faceIndex;

    printf("Successfully loaded model with %d vertices and %d faces.\n", vertexCount, faceIndex);

    return model;
}

void freeModel(Model* model) {
    free(model->vertices);
    free(model->faces);
    model->vertices = NULL;
    model->faces = NULL;
    model->vertexCount = 0;
    model->faceCount = 0;
}

int rayTriangleIntersection(Vertex rayOrigin, Vertex rayDirection, Vertex v0, Vertex v1, Vertex v2, float* t) {
    Vertex edge1, edge2, h, s, q;
    float a, f, u, v;

    edge1.x = v1.x - v0.x;
    edge1.y = v1.y - v0.y;
    edge1.z = v1.z - v0.z;

    edge2.x = v2.x - v0.x;
    edge2.y = v2.y - v0.y;
    edge2.z = v2.z - v0.z;

    h.x = rayDirection.y * edge2.z - rayDirection.z * edge2.y;
    h.y = rayDirection.z * edge2.x - rayDirection.x * edge2.z;
    h.z = rayDirection.x * edge2.y - rayDirection.y * edge2.x;

    a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

    if (a > -0.00001f && a < 0.00001f) {
        return 0;
    }

    f = 1.0f / a;
    s.x = rayOrigin.x - v0.x;
    s.y = rayOrigin.y - v0.y;
    s.z = rayOrigin.z - v0.z;

    u = f * (s.x * h.x + s.y * h.y + s.z * h.z);

    if (u < 0.0f || u > 1.0f) {
        return 0;
    }

    q.x = s.y * edge1.z - s.z * edge1.y;
    q.y = s.z * edge1.x - s.x * edge1.z;
    q.z = s.x * edge1.y - s.y * edge1.x;

    v = f * (rayDirection.x * q.x + rayDirection.y * q.y + rayDirection.z * q.z);

    if (v < 0.0f || u + v > 1.0f) {
        return 0;
    }

    *t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

    if (*t > 0.00001f) {
        return 1;
    }

    return 0;
}

int isInsideCarModel(int x, int y, int z, Model* model, int sizeX, int sizeY, int sizeZ) {
    float scaledX = (float)x / sizeX * 2.0f - 1.0f;
    float scaledY = (float)y / sizeY * 2.0f - 1.0f;
    float scaledZ = (float)z / sizeZ * 2.0f - 1.0f;

    Vertex rayOrigin = {scaledX, scaledY, -2.0f};
    Vertex rayDirection = {0.0f, 0.0f, 1.0f};

    int intersectionCount = 0;

    for (int i = 0; i < model->faceCount; i++) {
        // OBJ indices are 1-based
        int idx0 = model->faces[i].v1 - 1;
        int idx1 = model->faces[i].v2 - 1;
        int idx2 = model->faces[i].v3 - 1;
        
        if (idx0 < 0 || idx0 >= model->vertexCount ||
            idx1 < 0 || idx1 >= model->vertexCount ||
            idx2 < 0 || idx2 >= model->vertexCount) {
            continue;
        }
        
        Vertex v0 = model->vertices[idx0];
        Vertex v1 = model->vertices[idx1];
        Vertex v2 = model->vertices[idx2];

        float t;
        if (rayTriangleIntersection(rayOrigin, rayDirection, v0, v1, v2, &t)) {
            intersectionCount++;
        }
    }

    return (intersectionCount % 2 == 1);
}