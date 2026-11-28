#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>

// Solver checkpoint, a fixed header, tagged sections and a CRC-32 trailer.
// Native byte order, with a marker so another byte order is refused

#define CKPT_VERSION      1
#define CKPT_HEADER_BYTES 1024
#define CKPT_MAX_SECTIONS 32
#define CKPT_ERR_LEN      512

#define CKPT_TAG(a, b, c, d)                                                   \
    ((uint32_t)(a) | (uint32_t)(b) << 8 | (uint32_t)(c) << 16 |                \
     (uint32_t)(d) << 24)

typedef struct {
    uint32_t version;
    uint32_t floatBytes;
    char lattice[8]; // "D3Q19"
    char storage[8]; // "f-w"
    int32_t nx, ny, nz;
    char op[16]; // collision operator name
    int32_t smagorinsky;
    float smagorinskyCs;
    float tau;
    float u; // inlet velocity, lattice units
    int32_t spongeCells;
    float spongeTauMax;
    int32_t periodicX;
    int32_t periodicYZ;
    float turbulence;
    float bodyForce[3];
    uint64_t solidHash; // FNV-1a of the solid mask
    uint64_t qHash;     // FNV-1a of the Bouzidi q buffer
    int64_t step;       // LBM steps taken
    char model[512];
    char commit[48];
    int32_t dirty;
} CkptHeader;

typedef struct CkptWriter CkptWriter;
typedef struct CkptReader CkptReader;

// CRC-32 (IEEE, as zlib), start with crc = 0
uint32_t ckpt_crc32(uint32_t crc, const void *data, size_t n);

// FNV-1a 64, start with h = CKPT_FNV_OFFSET
#define CKPT_FNV_OFFSET 0xcbf29ce484222325ull
uint64_t ckpt_fnv1a(uint64_t h, const void *data, size_t n);

// Header with the format fields set and the rest zero
void ckpt_header_init(CkptHeader *h);

// Writes PATH.tmp, renamed to PATH by ckpt_commit. NULL on error
CkptWriter *
ckpt_create(const char *path, const CkptHeader *h, char *err, size_t errLen);
int ckpt_begin_section(CkptWriter *w, uint32_t tag, uint64_t bytes);
int ckpt_write(CkptWriter *w, const void *data, size_t bytes);
// Writes the trailer and renames, 0 on success. Frees w either way
int ckpt_commit(CkptWriter *w, char *err, size_t errLen);
// Drops the temp file and frees w
void ckpt_abort(CkptWriter *w);

// Checks format, sections and checksum before anything is read
CkptReader *
ckpt_open(const char *path, CkptHeader *h, char *err, size_t errLen);
// Positions the reader at a section, 0 if found
int ckpt_find(CkptReader *r, uint32_t tag, uint64_t *bytes);
int ckpt_read(CkptReader *r, void *dst, size_t bytes);
void ckpt_close(CkptReader *r);

// 0 if a checkpoint with header have can continue a run set up as want.
// The model path is not compared, the geometry hashes are
int ckpt_check(const CkptHeader *have,
               const CkptHeader *want,
               char *err,
               size_t errLen);

#endif // CHECKPOINT_H
