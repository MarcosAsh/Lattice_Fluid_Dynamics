#ifndef DRAG_METRICS_H
#define DRAG_METRICS_H

#define CD_HISTORY_SIZE 100
#define CD_SAMPLE_INTERVAL 20

// Run a DFT over the Cl time series to estimate the dominant
// shedding frequency, convert it into a Strouhal number and print
// the result. Called once after the render loop finishes.
void compute_strouhal(float *clSeries,
                      int clCount,
                      int lbmSubsteps,
                      float charLength,
                      float latticeVelocity);

#endif // DRAG_METRICS_H
