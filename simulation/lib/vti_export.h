#ifndef VTI_EXPORT_H
#define VTI_EXPORT_H

#include "lbm.h"

// Write LBM velocity, density (rho) and solid mask as a VTK ImageData
// (.vti) file at path/field_<step>.vti. Binary appended format.
void writeVTI(LBMGrid *grid, const char *path, int step);

#endif // VTI_EXPORT_H
