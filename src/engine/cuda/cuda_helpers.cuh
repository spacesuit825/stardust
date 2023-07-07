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

	// Template for launching kernels
	template<typename... Arguments>
	void KernelLaunch(std::string&& tag, int gs, int bs, void(*f)(Arguments...), Arguments... args) {
		f << <gs, bs >> > (args...);

		CUDA_ERR_CHECK(cudaPeekAtLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
	}
	
	__device__ inline float4 multiplyQuaternionsCUDA(
		float4 quaternion1, // (1, 0, 0, 0)
		float4 quaternion2
	)
	{
		float s0 = quaternion1.x;
		float s1 = quaternion2.x;

		float3 v0 = make_float3(quaternion1.y, quaternion1.z, quaternion1.w);
		float3 v1 = make_float3(quaternion2.y, quaternion2.z, quaternion2.w);

		float s2 = s0 * s1 - dot(v0, v1);
		float3 v2 = s0 * v1 + s1 * v0 + cross(v0, v1);

		return make_float4(s2, v2.x, v2.y, v2.z);
	}

	__device__ inline float4 multiplyQuaternionByVectorCUDA(
		float4 quaternion, // (1, 0, 0, 0)
		float4 vector
	)
	{
		float4 conjugate_quaternion = make_float4(
			quaternion.x,
			-quaternion.y,
			-quaternion.z,
			-quaternion.w);


		float4 quat_by_vec = multiplyQuaternionsCUDA(quaternion, vector);
		float4 rotated_vec = multiplyQuaternionsCUDA(quat_by_vec, conjugate_quaternion);

		return rotated_vec;
	}

	__device__ inline void prefixSumCUDA(
		uint32_t* values,
		unsigned int n)
	{
		int offset = 1;
		int a;
		uint32_t temp;

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

	__device__ inline void sumReduceCUDA(
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