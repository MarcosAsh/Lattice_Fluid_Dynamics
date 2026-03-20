#include "tensor.hpp"
#include "autodiff.hpp"
#include "loss.hpp"
#include "weights_io.hpp"
#include "layers/ad_linear.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(cond, msg)                                       \
    do {                                                       \
        tests_total++;                                         \
        if (!(cond)) {                                         \
            fprintf(stderr, "  FAIL: %s\n", msg);              \
        } else {                                               \
            printf("  PASS: %s\n", msg);                       \
            tests_passed++;                                    \
        }                                                      \
    } while (0)

static void test_mse_loss() {
    printf("test: mse_loss forward value\n");
    // pred = [1, 2, 3], target = [1, 2, 3] -> loss = 0
    Tensor p(3, 1), t(3, 1);
    p.data = {1.0f, 2.0f, 3.0f};
    t.data = {1.0f, 2.0f, 3.0f};
    auto pred = make_ad(p);
    auto targ = make_ad(t);
    auto loss = mse_loss(pred, targ);
    CHECK(std::fabs(loss->val.data[0]) < 1e-6f,
          "mse of identical vectors is 0");

    printf("test: mse_loss known value\n");
    // pred = [1, 0], target = [0, 0] -> loss = (1+0)/2 = 0.5
    Tensor p2(2, 1), t2(2, 1);
    p2.data = {1.0f, 0.0f};
    t2.data = {0.0f, 0.0f};
    auto pred2 = make_ad(p2);
    auto targ2 = make_ad(t2);
    auto loss2 = mse_loss(pred2, targ2);
    CHECK(std::fabs(loss2->val.data[0] - 0.5f) < 1e-6f,
          "mse([1,0],[0,0]) = 0.5");

    printf("test: mse_loss gradient\n");
    loss2->backward();
    // d/d pred = 2*(pred-target)/N = 2*[1,0]/2 = [1,0]
    CHECK(std::fabs(pred2->grad.data[0] - 1.0f) < 1e-5f,
          "grad[0] = 1.0");
    CHECK(std::fabs(pred2->grad.data[1]) < 1e-5f,
          "grad[1] = 0.0");
}

static void test_weights_io() {
    printf("test: save and load weights\n");
    clear_parameters();

    // Create a small network
    ADLinear layer(4, 2);

    // Grab a copy of the weights before saving
    auto& params = get_parameters();
    std::vector<float> w_orig(params[0]->val.data.begin(),
                              params[0]->val.data.end());
    std::vector<float> b_orig(params[1]->val.data.begin(),
                              params[1]->val.data.end());

    const char* path = "/tmp/test_lattice_weights.bin";
    CHECK(save_weights(path), "save_weights succeeds");

    // Corrupt the weights
    for (auto& v : params[0]->val.data) v = 999.0f;
    for (auto& v : params[1]->val.data) v = 999.0f;

    CHECK(load_weights(path), "load_weights succeeds");

    // Verify restoration
    bool match = true;
    for (size_t i = 0; i < w_orig.size(); ++i) {
        if (std::fabs(params[0]->val.data[i] - w_orig[i]) > 1e-7f)
            match = false;
    }
    for (size_t i = 0; i < b_orig.size(); ++i) {
        if (std::fabs(params[1]->val.data[i] - b_orig[i]) > 1e-7f)
            match = false;
    }
    CHECK(match, "loaded weights match original");

    // Clean up
    std::remove(path);
    clear_parameters();
}

int main() {
    printf("=== ML Framework Tests ===\n\n");
    test_mse_loss();
    test_weights_io();
    printf("\n=== Results: %d/%d passed ===\n",
           tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
