#include "optimizer.hpp"
#include <cmath>
#include "quantization.hpp"

SGD::SGD(float lr_) : lr(lr_) {}

void SGD::step() {
    auto& params = get_parameters();
    for (auto& p : params) {
        Tensor& val = p->val;
        Tensor& grad = p->grad;
        for (size_t i = 0, n = val.data.size(); i < n; ++i) {
            val.data[i] -= lr * grad.data[i];
        }
    }
    // Apply fake quantization after weight update if QAT enabled
    if (quant::g_qat_enabled) {
        for (auto& p : params) {
            quant::fake_quantize_inplace(p->val);
        }
    }
}

void SGD::zero_grad() {
    auto& params = get_parameters();
    for (auto& p : params) {
        p->grad.fill(0.0f);
    }
}

AdamW::AdamW(float lr_, float beta1_, float beta2_, float eps_, float weight_decay_, float clip_norm_)
    : lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_),
      weight_decay(weight_decay_), clip_norm(clip_norm_), t(0) {}

void AdamW::step() {
    auto& params = get_parameters();
    size_t Np = params.size();
    if (m.size() < Np) {
        for (size_t i = m.size(); i < Np; ++i) {
            int r = params[i]->val.rows;
            int c = params[i]->val.cols;
            m.emplace_back(r, c);
            m.back().fill(0.0f);
            v.emplace_back(r, c);
            v.back().fill(0.0f);
        }
    }
    if (clip_norm > 0.0f) {
        double sum_sq = 0.0;
        for (auto& p : params) {
            for (float g : p->grad.data) sum_sq += (double)g * g;
        }
        double norm = std::sqrt(sum_sq);
        if (norm > clip_norm) {
            double coef = clip_norm / (norm + 1e-6);
            for (auto& p : params) {
                for (auto& g : p->grad.data) g = (float)(g * coef);
            }
        }
    }
    t += 1;
    for (size_t i = 0; i < Np; ++i) {
        Tensor& grad = params[i]->grad;
        Tensor& val  = params[i]->val;
        Tensor& mi   = m[i];
        Tensor& vi   = v[i];
        size_t Sz = grad.data.size();
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        for (size_t j = 0; j < Sz; ++j) {
            double g = grad.data[j];
            mi.data[j] = beta1 * mi.data[j] + (1 - beta1) * g;
            vi.data[j] = beta2 * vi.data[j] + (1 - beta2) * g * g;
            double m_hat = mi.data[j] / bias_correction1;
            double v_hat = vi.data[j] / bias_correction2;
            double update = m_hat / (std::sqrt(v_hat) + eps);
            update += weight_decay * val.data[j];
            val.data[j] = (float)(val.data[j] - lr * update);
        }
    }
    // Apply fake quantization after weight update if QAT enabled
    if (quant::g_qat_enabled) {
        for (auto& p : params) {
            quant::fake_quantize_inplace(p->val);
        }
    }
}

void AdamW::zero_grad() {
    auto& params = get_parameters();
    for (auto& p : params) p->grad.fill(0.0f);
}