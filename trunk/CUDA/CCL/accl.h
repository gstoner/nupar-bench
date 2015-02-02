#ifndef SCCL_CU_H
#define SCCL_CU_H
void acclCuda(int *out, int *components, const int *in,
                 const uint nFrames, const int rows,
                 const int cols);

#endif
