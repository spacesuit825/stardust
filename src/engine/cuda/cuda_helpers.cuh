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

#define SQR(x) ((x) * (x))

namespace STARDUST {

	// Template for launching kernels
	template<typename... Arguments>
	void KernelLaunch(std::string&& tag, int gs, int bs, void(*f)(Arguments...), Arguments... args) {
		f << <gs, bs >> > (args...);

		CUDA_ERR_CHECK(cudaPeekAtLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
	}

	__device__ inline float getSignedFloat(float i) {
		if (i > 0.0) {
			return 1.0;
		}
		else if (i == 0.0) {
			return 1.0;
		}
		else {
			return -1.0;
		}
	}

	__device__ inline float4 convertVectorToSigns(float4 v) {
		float vin[4] = { v.x, v.y, v.z, v.w };
		float vout[4];
		for (int i = 0; i < 4; i++) {
			vout[i] = getSignedFloat(vin[i]);
		}

		return make_float4(vout[0], vout[1], vout[2], vout[3]);
	}

	__device__ inline float9 compute3x3Inverse(float9 m) {
		float det = m[0] * (m[4] * m[8] - m[7] * m[5]) -
			m[1] * (m[3] * m[8] - m[5] * m[6]) +
			m[2] * (m[3] * m[7] - m[4] * m[6]);

		if (det == 0.0f) {
			return m;
		}

		float invdet = 1 / det;

		float9 minv = {
			(m[4] * m[8] - m[7] * m[5])* invdet,
			(m[2] * m[7] - m[1] * m[8])* invdet,
			(m[1] * m[5] - m[2] * m[4])* invdet,
			(m[5] * m[6] - m[3] * m[8])* invdet,
			(m[0] * m[8] - m[2] * m[6])* invdet,
			(m[3] * m[2] - m[0] * m[5])* invdet,
			(m[3] * m[7] - m[6] * m[4])* invdet,
			(m[6] * m[1] - m[0] * m[7])* invdet,
			(m[0] * m[4] - m[3] * m[1])* invdet
		};

		return minv;
	}

	__device__ inline float9 computeTranspose(float9 m) {
		float9 mtrans = {
			m[0],
			m[3],
			m[6],
			m[1],
			m[4],
			m[7],
			m[2],
			m[5],
			m[8]
		};

		return mtrans;
	}

	__device__ inline float9 computeMatrixMultiplication(float9 m1, float9 m2) {
		float9 mmulti;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				int sum = 0;
				for (int k = 0; k < 3; k++)
					sum = sum + m1[i * 3 + k] * m2[k * 3 + j];
				mmulti[i * 3 + j] = sum;
			}
		}

		return mmulti;
	}

	__device__ inline float4 computeMatrixVectorMultiplication(float9 m, float4 v) {
		float vec[3] = { v.x, v.y, v.z };
		float mmultiv[3] = { 0.0f, 0.0f, 0.0f };
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				mmultiv[i] += m[3 * i + j] * vec[j];
			}
		}

		return make_float4(mmultiv[0], mmultiv[1], mmultiv[2], 0.0f);
	}

	__device__ inline float9 quatToRotationCUDA(float4 quat) {
		// Quat: {s, vx, vy, vz}
		float9 rotation_matrix = {
			1 - 2 * SQR(quat.z) - 2 * SQR(quat.w),
			2 * quat.y * quat.z - 2 * quat.x * quat.w,
			2 * quat.y * quat.w + 2 * quat.x * quat.z,
			2 * quat.y * quat.z + 2 * quat.x * quat.w,
			1 - 2 * SQR(quat.y) - 2 * SQR(quat.w),
			2 * quat.z * quat.w - 2 * quat.x * quat.y,
			2 * quat.y * quat.w - 2 * quat.x * quat.z,
			2 * quat.z * quat.w + 2 * quat.x * quat.y,
			1 - 2 * SQR(quat.y) - 2 * SQR(quat.z) };

		return rotation_matrix;
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