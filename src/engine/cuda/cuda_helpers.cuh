// C++
#include <string>
#include <iostream>
#include <fstream>

// Internal
#include "../types.hpp"
#include "../helper_math.hpp"

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

	__device__ inline void debugVector(float4 v) {
		printf("debug vector: \n");
		printf("%.3f, %.3f, %.3f, %.3f\n", v.x, v.y, v.z, v.w);
	}

	__device__ inline void debug3x3Matrix(float9 m) {
		printf("debug matrix: \n");
		printf("%.3f, %.3f, %.3f\n %.3f, %.3f, %.3f\n %.3f, %.3f, %.3f\n", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);

		return;
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
		float9 minv;
		float cofactor11 = m[4] * m[8] - m[7] * m[5], cofactor12 = m[7] * m[2] - m[1] * m[8], cofactor13 = m[1] * m[5] - m[4] * m[2];
		float determinant = m[0] * cofactor11 + m[3] * cofactor12 + m[6] * cofactor13;
		if (1e-4 > determinant) {
			return m;
		}
		float s = 1 / determinant;
		minv[0] = s * cofactor11; minv[1] = s * cofactor12; minv[2] = s * cofactor13;
		minv[3] = s * m[6] * m[5] - s * m[3] * m[8]; minv[4] = s * m[0] * m[8] - s * m[6] * m[2]; minv[5] = s * m[3] * m[2] - s * m[0] * m[5];
		minv[6] = s * m[3] * m[7] - s * m[6] * m[4]; minv[7] = s * m[6] * m[1] - s * m[0] * m[7]; minv[8] = s * m[0] * m[4] - s * m[3] * m[1];

		return minv;
	}

	__device__ inline float9 computeTranspose(float9 m) {
		float9 mtrans;
		mtrans[0] = m[0]; mtrans[1] = m[3]; mtrans[2] = m[6];
		mtrans[3] = m[1]; mtrans[4] = m[4]; mtrans[5] = m[7];
		mtrans[6] = m[2]; mtrans[7] = m[5]; mtrans[8] = m[8];

		return mtrans;
	}

	__device__ inline float9 computeMatrixMultiplication(float9 m1, float9 m2) {
		float9 mmulti;
		mmulti[0] = m1[0] * m2[0] + m1[3] * m2[1] + m1[6] * m2[2];
		mmulti[1] = m1[1] * m2[0] + m1[4] * m2[1] + m1[7] * m2[2];
		mmulti[2] = m1[2] * m2[0] + m1[5] * m2[1] + m1[8] * m2[2];
		mmulti[3] = m1[0] * m2[3] + m1[3] * m2[4] + m1[6] * m2[5];
		mmulti[4] = m1[1] * m2[3] + m1[4] * m2[4] + m1[7] * m2[5];
		mmulti[5] = m1[2] * m2[3] + m1[5] * m2[4] + m1[8] * m2[5];
		mmulti[6] = m1[0] * m2[6] + m1[3] * m2[7] + m1[6] * m2[8];
		mmulti[7] = m1[1] * m2[6] + m1[4] * m2[7] + m1[7] * m2[8];
		mmulti[8] = m1[2] * m2[6] + m1[5] * m2[7] + m1[8] * m2[8];

		return mmulti;
	}

	__device__ inline float4 computeMatrixVectorMultiplication(float9 m, float4 v) {
		float4 mvec;
		mvec.x = m[0] * v.x + m[3] * v.y + m[6] * v.z;
		mvec.y = m[1] * v.x + m[4] * v.y + m[7] * v.z;
		mvec.z = m[2] * v.x + m[5] * v.y + m[8] * v.z;
		mvec.w = 0.0f;

		return mvec;
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
		// Convert vector to pseudo-quaternion
		float4 vector_quaternion = make_float4(
			0.0f,
			vector.x,
			vector.y,
			vector.z
		);

		float4 conjugate_quaternion = make_float4(
			quaternion.x,
			-quaternion.y,
			-quaternion.z,
			-quaternion.w);


		float4 quat_by_vec = multiplyQuaternionsCUDA(quaternion, vector_quaternion);
		float4 rotated_vec = multiplyQuaternionsCUDA(quat_by_vec, conjugate_quaternion);

		return make_float4(rotated_vec.y, rotated_vec.z, rotated_vec.w, 0.0f);
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