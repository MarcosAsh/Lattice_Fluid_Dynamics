#ifndef DRAG_METRICS_H
#define DRAG_METRICS_H

#define CD_HISTORY_SIZE    100
#define CD_SAMPLE_INTERVAL 20

// Run a DFT over the Cl time series to estimate the dominant
// shedding frequency, convert it into a Strouhal number and print
// the result. Called once after the render loop finishes. Returns 1 and
// fills st/freq (either may be NULL) when the series was long enough.
int compute_strouhal(float *clSeries,
                     int clCount,
                     int lbmSubsteps,
                     float charLength,
                     float latticeVelocity,
                     float *st,
                     float *freq);

#endif // DRAG_METRICS_H
