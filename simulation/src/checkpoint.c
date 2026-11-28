#include "../lib/checkpoint.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#endif

static const char kMagic[8] = {'L', 'A', 'T', 'C', 'K', 'P', 'T', '\0'};
#define ENDIAN_MARK 0x01020304u
#define END_TAG     CKPT_TAG('E', 'N', 'D', '!')
#define IO_CHUNK    (8u << 20)

struct CkptWriter {
    FILE *f;
    char *path;
    char *tmp;
    uint32_t crc;
    uint64_t left; // bytes still owed to the open section
    int open;
    int failed;
};

typedef struct {
    uint32_t tag;
    uint64_t offset; // first data byte
    uint64_t bytes;
} Section;

struct CkptReader {
    FILE *f;
    Section sec[CKPT_MAX_SECTIONS];
    int nsec;
    uint64_t left;
};

static void set_err(char *err, size_t n, const char *fmt, ...) {
    if (!err || n == 0)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, n, fmt, ap);
    va_end(ap);
}

static uint32_t crcTable[8][256];
static int crcReady;

static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crcTable[0][i] = c;
    }
    for (int t = 1; t < 8; t++)
        for (int i = 0; i < 256; i++)
            crcTable[t][i] = (crcTable[t - 1][i] >> 8) ^
                             crcTable[0][crcTable[t - 1][i] & 0xff];
    crcReady = 1;
}

// Slicing by 8 over single bytes, so host byte order does not matter
uint32_t ckpt_crc32(uint32_t crc, const void *data, size_t n) {
    if (!crcReady)
        crc_init();
    const unsigned char *p = (const unsigned char *)data;
    crc = ~crc;
    while (n >= 8) {
        uint32_t lo = crc ^ ((uint32_t)p[0] | (uint32_t)p[1] << 8 |
                             (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24);
        crc = crcTable[7][lo & 0xff] ^ crcTable[6][(lo >> 8) & 0xff] ^
              crcTable[5][(lo >> 16) & 0xff] ^ crcTable[4][lo >> 24] ^
              crcTable[3][p[4]] ^ crcTable[2][p[5]] ^ crcTable[1][p[6]] ^
              crcTable[0][p[7]];
        p += 8;
        n -= 8;
    }
    while (n--)
        crc = crcTable[0][(crc ^ *p++) & 0xff] ^ (crc >> 8);
    return ~crc;
}

uint64_t ckpt_fnv1a(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

void ckpt_header_init(CkptHeader *h) {
    memset(h, 0, sizeof(*h));
    h->version = CKPT_VERSION;
    h->floatBytes = sizeof(float);
    snprintf(h->lattice, sizeof(h->lattice), "D3Q19");
    snprintf(h->storage, sizeof(h->storage), "f-w");
}

// Fixed header layout, one field after another from offset 0
typedef struct {
    unsigned char *b;
    size_t pos;
    int write;
} Cursor;

static void field(Cursor *c, void *v, size_t n) {
    if (c->pos + n > CKPT_HEADER_BYTES)
        return;
    if (c->write)
        memcpy(c->b + c->pos, v, n);
    else
        memcpy(v, c->b + c->pos, n);
    c->pos += n;
}

static void header_fields(Cursor *c, CkptHeader *h) {
    field(c, h->lattice, sizeof(h->lattice));
    field(c, h->storage, sizeof(h->storage));
    field(c, &h->nx, 4);
    field(c, &h->ny, 4);
    field(c, &h->nz, 4);
    field(c, h->op, sizeof(h->op));
    field(c, &h->smagorinsky, 4);
    field(c, &h->smagorinskyCs, 4);
    field(c, &h->tau, 4);
    field(c, &h->u, 4);
    field(c, &h->spongeCells, 4);
    field(c, &h->spongeTauMax, 4);
    field(c, &h->periodicX, 4);
    field(c, &h->periodicYZ, 4);
    field(c, &h->turbulence, 4);
    field(c, h->bodyForce, sizeof(h->bodyForce));
    field(c, &h->solidHash, 8);
    field(c, &h->qHash, 8);
    field(c, &h->step, 8);
    field(c, h->model, sizeof(h->model));
    field(c, h->commit, sizeof(h->commit));
    field(c, &h->dirty, 4);
}

static void pack_header(const CkptHeader *src, unsigned char *b) {
    CkptHeader h = *src;
    memset(b, 0, CKPT_HEADER_BYTES);
    Cursor c = {b, 0, 1};
    uint32_t mark = ENDIAN_MARK, hb = CKPT_HEADER_BYTES;
    field(&c, (void *)kMagic, sizeof(kMagic));
    field(&c, &h.version, 4);
    field(&c, &mark, 4);
    field(&c, &hb, 4);
    field(&c, &h.floatBytes, 4);
    // Strings always end in a zero byte
    h.lattice[sizeof(h.lattice) - 1] = h.storage[sizeof(h.storage) - 1] = 0;
    h.op[sizeof(h.op) - 1] = h.model[sizeof(h.model) - 1] = 0;
    h.commit[sizeof(h.commit) - 1] = 0;
    header_fields(&c, &h);
}

static uint32_t bswap32(uint32_t v) {
    return (v >> 24) | ((v >> 8) & 0xff00) | ((v << 8) & 0xff0000) | (v << 24);
}

static int
unpack_header(const unsigned char *b, CkptHeader *h, char *err, size_t n) {
    if (memcmp(b, kMagic, sizeof(kMagic)) != 0) {
        set_err(err, n, "not a Lattice checkpoint (bad magic)");
        return -1;
    }
    Cursor c = {(unsigned char *)b, sizeof(kMagic), 0};
    uint32_t mark, hb;
    memset(h, 0, sizeof(*h));
    field(&c, &h->version, 4);
    field(&c, &mark, 4);
    field(&c, &hb, 4);
    field(&c, &h->floatBytes, 4);
    if (mark != ENDIAN_MARK) {
        set_err(err,
                n,
                mark == bswap32(ENDIAN_MARK)
                    ? "checkpoint byte order differs from this machine"
                    : "bad byte order marker 0x%08" PRIx32,
                mark);
        return -1;
    }
    if (h->version != CKPT_VERSION) {
        set_err(err,
                n,
                "checkpoint format version %" PRIu32 ", this build reads %d",
                h->version,
                CKPT_VERSION);
        return -1;
    }
    if (hb != CKPT_HEADER_BYTES) {
        set_err(err, n, "bad header size %" PRIu32, hb);
        return -1;
    }
    if (h->floatBytes != sizeof(float)) {
        set_err(err,
                n,
                "checkpoint floats are %" PRIu32 " bytes, expected %zu",
                h->floatBytes,
                sizeof(float));
        return -1;
    }
    header_fields(&c, h);
    h->lattice[sizeof(h->lattice) - 1] = h->storage[sizeof(h->storage) - 1] = 0;
    h->op[sizeof(h->op) - 1] = h->model[sizeof(h->model) - 1] = 0;
    h->commit[sizeof(h->commit) - 1] = 0;
    return 0;
}

static int put(CkptWriter *w, const void *data, size_t n) {
    if (w->failed)
        return -1;
    if (n && fwrite(data, 1, n, w->f) != n) {
        w->failed = 1;
        return -1;
    }
    w->crc = ckpt_crc32(w->crc, data, n);
    return 0;
}

CkptWriter *
ckpt_create(const char *path, const CkptHeader *h, char *err, size_t errLen) {
    CkptWriter *w = (CkptWriter *)calloc(1, sizeof(*w));
    size_t len = strlen(path);
    if (w) {
        w->path = (char *)malloc(len + 1);
        w->tmp = (char *)malloc(len + 5);
    }
    if (!w || !w->path || !w->tmp) {
        set_err(err, errLen, "out of memory");
        if (w) {
            free(w->path);
            free(w->tmp);
        }
        free(w);
        return NULL;
    }
    memcpy(w->path, path, len + 1);
    snprintf(w->tmp, len + 5, "%s.tmp", path);
    w->f = fopen(w->tmp, "wb");
    if (!w->f) {
        set_err(err, errLen, "cannot create %s: %s", w->tmp, strerror(errno));
        free(w->path);
        free(w->tmp);
        free(w);
        return NULL;
    }
    unsigned char b[CKPT_HEADER_BYTES];
    pack_header(h, b);
    if (put(w, b, sizeof(b)) != 0) {
        set_err(err, errLen, "cannot write %s: %s", w->tmp, strerror(errno));
        ckpt_abort(w);
        return NULL;
    }
    return w;
}

int ckpt_begin_section(CkptWriter *w, uint32_t tag, uint64_t bytes) {
    if (w->open && w->left != 0)
        w->failed = 1;
    uint32_t head[2] = {tag, 0};
    put(w, head, sizeof(head));
    put(w, &bytes, sizeof(bytes));
    w->open = 1;
    w->left = bytes;
    return w->failed ? -1 : 0;
}

int ckpt_write(CkptWriter *w, const void *data, size_t bytes) {
    if (!w->open || bytes > w->left) {
        w->failed = 1;
        return -1;
    }
    w->left -= bytes;
    return put(w, data, bytes);
}

void ckpt_abort(CkptWriter *w) {
    if (!w)
        return;
    if (w->f)
        fclose(w->f);
    remove(w->tmp);
    free(w->path);
    free(w->tmp);
    free(w);
}

int ckpt_commit(CkptWriter *w, char *err, size_t errLen) {
    if (w->open && w->left != 0) {
        set_err(err,
                errLen,
                "section short by %" PRIu64 " bytes",
                (uint64_t)w->left);
        ckpt_abort(w);
        return -1;
    }
    uint32_t tag = END_TAG;
    put(w, &tag, sizeof(tag));
    uint32_t crc = w->crc;
    int ok = !w->failed && fwrite(&crc, 1, sizeof(crc), w->f) == sizeof(crc);
    ok = ok && fflush(w->f) == 0;
#ifndef _WIN32
    ok = ok && fsync(fileno(w->f)) == 0;
#endif
    int e = errno;
    ok = (fclose(w->f) == 0) && ok;
    w->f = NULL;
    if (ok && rename(w->tmp, w->path) != 0) {
        e = errno;
        ok = 0;
    }
    if (!ok) {
        set_err(err, errLen, "cannot write %s: %s", w->tmp, strerror(e));
        ckpt_abort(w);
        return -1;
    }
    free(w->path);
    free(w->tmp);
    free(w);
    return 0;
}

static void tag_name(uint32_t tag, char out[5]) {
    for (int i = 0; i < 4; i++) {
        char ch = (char)((tag >> (8 * i)) & 0xff);
        out[i] = (ch >= 32 && ch < 127) ? ch : '?';
    }
    out[4] = 0;
}

CkptReader *
ckpt_open(const char *path, CkptHeader *h, char *err, size_t errLen) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        set_err(err, errLen, "cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    CkptReader *r = (CkptReader *)calloc(1, sizeof(*r));
    unsigned char *buf = (unsigned char *)malloc(IO_CHUNK);
    if (!r || !buf) {
        set_err(err, errLen, "out of memory");
        goto fail;
    }
    r->f = f;
    if (fseeko(f, 0, SEEK_END) != 0) {
        set_err(err, errLen, "cannot seek %s", path);
        goto fail;
    }
    uint64_t size = (uint64_t)ftello(f);
    rewind(f);

    size_t got = fread(buf, 1, CKPT_HEADER_BYTES, f);
    if (got >= sizeof(kMagic) && memcmp(buf, kMagic, sizeof(kMagic)) != 0) {
        set_err(
            err, errLen, "%s is not a Lattice checkpoint (bad magic)", path);
        goto fail;
    }
    if (got < CKPT_HEADER_BYTES) {
        set_err(err,
                errLen,
                "checkpoint truncated: %" PRIu64 " bytes, header needs %d",
                size,
                CKPT_HEADER_BYTES);
        goto fail;
    }
    if (unpack_header(buf, h, err, errLen) != 0)
        goto fail;

    // Section walk, every length must fit before the end marker
    uint64_t pos = CKPT_HEADER_BYTES;
    for (;;) {
        uint32_t tag;
        if (pos + 8 > size) {
            set_err(err,
                    errLen,
                    "checkpoint truncated: no end marker after %" PRIu64
                    " bytes",
                    size);
            goto fail;
        }
        if (fseeko(f, (off_t)pos, SEEK_SET) != 0 || fread(&tag, 4, 1, f) != 1) {
            set_err(err, errLen, "cannot read %s", path);
            goto fail;
        }
        if (tag == END_TAG) {
            if (pos + 8 != size) {
                set_err(err,
                        errLen,
                        "checkpoint has %" PRIu64 " bytes after the end marker",
                        size - pos - 8);
                goto fail;
            }
            break;
        }
        char name[5];
        tag_name(tag, name);
        uint64_t bytes;
        uint32_t pad;
        if (pos + 16 > size || fread(&pad, 4, 1, f) != 1 ||
            fread(&bytes, 8, 1, f) != 1) {
            set_err(err,
                    errLen,
                    "checkpoint truncated in the %s section header",
                    name);
            goto fail;
        }
        if (bytes > size - pos - 16) {
            set_err(err,
                    errLen,
                    "checkpoint truncated: section %s needs %" PRIu64
                    " bytes, %" PRIu64 " left",
                    name,
                    bytes,
                    size - pos - 16);
            goto fail;
        }
        if (r->nsec == CKPT_MAX_SECTIONS) {
            set_err(err,
                    errLen,
                    "checkpoint has more than %d sections",
                    CKPT_MAX_SECTIONS);
            goto fail;
        }
        r->sec[r->nsec++] = (Section){tag, pos + 16, bytes};
        pos += 16 + bytes;
    }

    // Checksum over everything but the stored value
    rewind(f);
    uint32_t crc = 0;
    uint64_t todo = size - 4;
    while (todo > 0) {
        size_t n = todo < IO_CHUNK ? (size_t)todo : IO_CHUNK;
        if (fread(buf, 1, n, f) != n) {
            set_err(err, errLen, "cannot read %s", path);
            goto fail;
        }
        crc = ckpt_crc32(crc, buf, n);
        todo -= n;
    }
    uint32_t stored;
    if (fread(&stored, 4, 1, f) != 1) {
        set_err(err, errLen, "cannot read %s", path);
        goto fail;
    }
    if (stored != crc) {
        set_err(err,
                errLen,
                "checkpoint checksum mismatch (stored %08" PRIx32
                ", computed %08" PRIx32 ")",
                stored,
                crc);
        goto fail;
    }
    free(buf);
    return r;

fail:
    free(buf);
    free(r);
    fclose(f);
    return NULL;
}

int ckpt_find(CkptReader *r, uint32_t tag, uint64_t *bytes) {
    for (int i = 0; i < r->nsec; i++) {
        if (r->sec[i].tag != tag)
            continue;
        if (fseeko(r->f, (off_t)r->sec[i].offset, SEEK_SET) != 0)
            return -1;
        r->left = r->sec[i].bytes;
        if (bytes)
            *bytes = r->sec[i].bytes;
        return 0;
    }
    return -1;
}

int ckpt_read(CkptReader *r, void *dst, size_t bytes) {
    if (bytes > r->left)
        return -1;
    if (bytes && fread(dst, 1, bytes, r->f) != bytes)
        return -1;
    r->left -= bytes;
    return 0;
}

void ckpt_close(CkptReader *r) {
    if (!r)
        return;
    fclose(r->f);
    free(r);
}

static int same_bits(float a, float b) {
    return memcmp(&a, &b, sizeof(float)) == 0;
}

#define FLOAT_FIELD(name, label)                                               \
    if (!same_bits(have->name, want->name)) {                                  \
        set_err(err,                                                           \
                n,                                                             \
                "checkpoint " label " %.9g does not match %.9g",               \
                (double)have->name,                                            \
                (double)want->name);                                           \
        return -1;                                                             \
    }

#define INT_FIELD(name, label)                                                 \
    if (have->name != want->name) {                                            \
        set_err(err,                                                           \
                n,                                                             \
                "checkpoint " label " %d does not match %d",                   \
                (int)have->name,                                               \
                (int)want->name);                                              \
        return -1;                                                             \
    }

int ckpt_check(const CkptHeader *have,
               const CkptHeader *want,
               char *err,
               size_t n) {
    if (strcmp(have->lattice, want->lattice) != 0 ||
        strcmp(have->storage, want->storage) != 0) {
        set_err(err,
                n,
                "checkpoint lattice %s storage %s, expected %s %s",
                have->lattice,
                have->storage,
                want->lattice,
                want->storage);
        return -1;
    }
    if (have->nx != want->nx || have->ny != want->ny || have->nz != want->nz) {
        set_err(err,
                n,
                "checkpoint grid %dx%dx%d does not match %dx%dx%d",
                have->nx,
                have->ny,
                have->nz,
                want->nx,
                want->ny,
                want->nz);
        return -1;
    }
    if (strcmp(have->op, want->op) != 0) {
        set_err(err,
                n,
                "checkpoint collision operator %s does not match %s",
                have->op,
                want->op);
        return -1;
    }
    FLOAT_FIELD(tau, "tau")
    FLOAT_FIELD(u, "inlet velocity")
    INT_FIELD(smagorinsky, "Smagorinsky flag")
    if (want->smagorinsky)
        FLOAT_FIELD(smagorinskyCs, "Smagorinsky constant")
    INT_FIELD(spongeCells, "sponge width")
    FLOAT_FIELD(spongeTauMax, "sponge tau")
    INT_FIELD(periodicX, "periodic x flag")
    INT_FIELD(periodicYZ, "periodic y/z flag")
    FLOAT_FIELD(turbulence, "inlet turbulence")
    for (int i = 0; i < 3; i++)
        FLOAT_FIELD(bodyForce[i], "body force")
    if (have->solidHash != want->solidHash) {
        set_err(err,
                n,
                "checkpoint solid mask %016" PRIx64
                " does not match %016" PRIx64
                " (model %s in the checkpoint, %s now)",
                have->solidHash,
                want->solidHash,
                have->model,
                want->model);
        return -1;
    }
    if (have->qHash != want->qHash) {
        set_err(err,
                n,
                "checkpoint wall distances %016" PRIx64
                " do not match %016" PRIx64 " (model %s in the checkpoint)",
                have->qHash,
                want->qHash,
                have->model);
        return -1;
    }
    return 0;
}
