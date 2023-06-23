#ifndef _STARDUST_CUDAUTILS_HEADER_
#define _STARDUST_CUDAUTILS_HEADER_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_ERR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); };
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif // _STARDUST_CUDAUTILS_HEADER_