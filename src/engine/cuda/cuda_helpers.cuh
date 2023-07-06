// C++
#include <string>
#include <iostream>
#include <fstream>

// CUDA
#include <vector_types.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvfunctional>
#include <device_functions.h>

namespace STARDUST {
	
	__device__ void prefixSumCUDA(
		int* values,
		unsigned int n)
	{
		int offset = 1;
		int a;
		int temp;

		for (int d = n / 2; d; d /= 2) {
			
			__syncthreads();

			if (threadIdx.x < d) {
				a = (threadIdx.x * 2 + 1) * offset - 1;
				values[a + offset] += values[a];
			}

			offset *= 2;
		}

		if (!threadIdx.x) {
			values[n - 1] = 0;
		}

		for (int d = 1; d < n; d *= 2) {

			__syncthreads();
			offset /= 2;

			if (threadIdx.x < d) {
				a = (threadIdx.x * 2 + 1) * offset - 1;
				temp = values[a];
				values[a] = values[a + offset];
				values[a + offset] += temp;
			}
		}
	}

	__device__ void sumReduceCUDA(
		unsigned int* values, 
		unsigned int* out) 
	{
		__syncthreads();

		unsigned int threads = blockDim.x;
		unsigned int half = threads / 2;

		while (half) {
			if (threadIdx.x < half) {
				for (int k = threadIdx.x + half; k < threads; k += half) {
					values[threadIdx.x] += values[k];
				}

				threads = half;
			}

			half /= 2;

			__syncthreads();
		}

		if (!threadIdx.x) {
			atomicAdd(out, values[0]);
		}
	}

}