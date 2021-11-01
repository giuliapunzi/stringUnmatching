#pragma once
#include <cstdio>

#ifdef DEBUG
#define CUDA_CHECK( call ) {                        \
    const cudaError_t error = call;                 \
    if (error != cudaSuccess) {                     \
        printf("Error in %s, line %d: %s",          \
            __FILE__, __LINE__,                     \
            cudaGetErrorString(error));             \
        exit(-error);                               \
    }                                               \
}
#else
#define CUDA_CHECK( call ) call;
#endif
