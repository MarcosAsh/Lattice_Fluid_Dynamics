#ifndef ML_PREDICT_H
#define ML_PREDICT_H

// Lightweight C inference for the Cd/Cl surrogate model.
// No dependency on the C++ ML framework -- just loads a model.bin
// and model_norm.bin produced by ml/train, then runs the forward pass.

typedef struct MLPredictor MLPredictor;

// Load model weights and normalizer stats from disk.
// Returns NULL on failure.
MLPredictor *ML_Load(const char *weights_path, const char *norm_path);

// Free all memory held by the predictor.
void ML_Free(MLPredictor *m);

// Run a single forward pass.
// Inputs are in physical units (normalization is applied internally).
// Writes predicted drag and lift coefficients into *cd and *cl.
void ML_Predict(MLPredictor *m,
                float wind_speed,
                float reynolds,
                float model_id,
                float *cd,
                float *cl);

#endif // ML_PREDICT_H
