#define _USE_MATH_DEFINES
#include "../lib/drag_metrics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void compute_strouhal(float *clSeries,
                      int clCount,
                      int lbmSubsteps,
                      float charLength,
                      float latticeVelocity) {
    // Strouhal number extraction from Cl time series.
    // St = f_peak * L / U, where f_peak is the dominant shedding
    // frequency found via DFT of the Cl signal.
    if (!clSeries || clCount < 12) {
        return;
    }

    // Discard first 35% as transient
    int start = clCount * 35 / 100;
    int n = clCount - start;
    float *sig = clSeries + start;

    // Remove mean
    float mean = 0;
    for (int i = 0; i < n; i++)
        mean += sig[i];
    mean /= n;

    // Hann window + mean removal
    float *win = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
        win[i] = (sig[i] - mean) * w;
    }

    // DFT over one-sided spectrum (skip DC).
    // Each Cl sample corresponds to CD_SAMPLE_INTERVAL frames,
    // each frame running lbmSubsteps lattice timesteps.
    float fs = 1.0f / (CD_SAMPLE_INTERVAL * lbmSubsteps);
    int nfreqs = n / 2;
    float peakPow = 0;
    float peakFreq = 0;
    for (int k = 1; k < nfreqs; k++) {
        float re = 0, im = 0;
        for (int i = 0; i < n; i++) {
            float angle = 2.0f * (float)M_PI * k * i / n;
            re += win[i] * cosf(angle);
            im -= win[i] * sinf(angle);
        }
        float pw = re * re + im * im;
        if (pw > peakPow) {
            peakPow = pw;
            peakFreq = (float)k * fs / n;
        }
    }

    float charL = charLength; // lattice units, set during LBM init
    float st =
        (latticeVelocity > 1e-10f) ? peakFreq * charL / latticeVelocity : 0;
    printf("St=%.4f (f=%.6f, L=%.1f, U=%.3f)\n",
           st,
           peakFreq,
           charL,
           latticeVelocity);
    free(win);
}
