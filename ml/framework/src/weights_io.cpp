#include "weights_io.hpp"
#include <cstdint>
#include <cstdio>

static constexpr uint32_t MAGIC = 0x4C545753; // "LTWS"
static constexpr uint32_t VERSION = 1;

bool save_weights(const std::string& path) {
    auto& params = get_parameters();
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    uint32_t magic = MAGIC;
    uint32_t version = VERSION;
    uint32_t count = static_cast<uint32_t>(params.size());
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&count, 4, 1, f);

    for (auto& p : params) {
        uint32_t ndim = static_cast<uint32_t>(p->val.shape.size());
        fwrite(&ndim, 4, 1, f);
        for (int d : p->val.shape) {
            uint32_t s = static_cast<uint32_t>(d);
            fwrite(&s, 4, 1, f);
        }
        fwrite(p->val.data.data(),
               sizeof(float),
               p->val.data.size(),
               f);
    }

    fclose(f);
    return true;
}

bool load_weights(const std::string& path) {
    auto& params = get_parameters();
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic, version, count;
    if (fread(&magic, 4, 1, f) != 1 || magic != MAGIC) {
        fclose(f);
        return false;
    }
    if (fread(&version, 4, 1, f) != 1 || version != VERSION) {
        fclose(f);
        return false;
    }
    if (fread(&count, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    if (count != static_cast<uint32_t>(params.size())) {
        fprintf(stderr,
                "weights_io: param count mismatch (file=%u, model=%zu)\n",
                count, params.size());
        fclose(f);
        return false;
    }

    for (auto& p : params) {
        uint32_t ndim;
        if (fread(&ndim, 4, 1, f) != 1) { fclose(f); return false; }

        // Validate shape matches
        if (ndim != static_cast<uint32_t>(p->val.shape.size())) {
            fprintf(stderr, "weights_io: ndim mismatch\n");
            fclose(f);
            return false;
        }
        for (int d = 0; d < static_cast<int>(ndim); ++d) {
            uint32_t s;
            if (fread(&s, 4, 1, f) != 1) { fclose(f); return false; }
            if (static_cast<int>(s) != p->val.shape[d]) {
                fprintf(stderr,
                        "weights_io: shape mismatch at dim %d "
                        "(file=%u, model=%d)\n",
                        d, s, p->val.shape[d]);
                fclose(f);
                return false;
            }
        }

        size_t n = p->val.data.size();
        if (fread(p->val.data.data(), sizeof(float), n, f) != n) {
            fclose(f);
            return false;
        }
    }

    fclose(f);
    return true;
}
