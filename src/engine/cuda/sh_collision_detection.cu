#ifndef CUDACC
#define CUDACC
#endif

// Internal
#include "../engine.hpp"
#include "cuda_utils.hpp"
#include "collision_detection.cuh"
#include "cuda_helpers.cuh"

// C++
#include <string>
#include <iostream>
#include <fstream>

// CUDA
#include <vector_types.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvfunctional>
#include <cuda_runtime_api.h>


namespace STARDUST {

	__global__ void constructCollisionListCUDA(
		unsigned int n_spheres,
		float cell_dim, // Max cell size in any dimension
		uint32_t* cells,
		uint32_t* spheres,
		float* sphere_sizes,
		float4* sphere_positions,
		unsigned int* d_temp_ptr)//,
		//unsigned int* cell_count)
	{
		extern __shared__ unsigned int t[];

		unsigned int count = 0;

		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		// If more spheres that threads, iterate across the remaining particles
		for (int i = idx; i < n_spheres; i += gridDim.x * blockDim.x) {

			uint32_t hash = 0;
			unsigned int sides = 0;

			int h = i * 9;
			int m = 1;

			int q;
			int r;
			float x;
			float a;

			float4 position = sphere_positions[i];
			float pos[3] = { position.x, position.y, position.z };

			for (int j = 0; j < 3; j++) {

				x = pos[j];

				hash = hash << 8 | (uint32_t)(x / cell_dim);

				x -= floor(x / cell_dim) * cell_dim;

				a = sphere_sizes[i];
				sides <<= 2;

				if (x < a) {
					sides |= 3;
				}
				else if (cell_dim - x < a) {
					sides |= 1;
				}
			}

			cells[h] = hash << 1 | 0x00;
			spheres[h] = i << 1 | 0x01;
			count++;

			for (int j = 0; j < 27; j++) {
				if (j == 27 / 2) {
					continue;
				}

				q = j;
				hash = 0;

				for (int k = 0; k < 3; k++) {
					r = q % 3 - 1;
					x = pos[k];

					if (r && (sides >> (3 - k - 1) * 2 & 0x03 ^ r) ||
						x + r * cell_dim < 0 || x + r * cell_dim >= 1) {

						hash = 0xffffffff; // UINT32MAX
						break;
					}

					hash = hash << 8 | (uint32_t)(x / cell_dim) + r;
					q /= 3;
				}

				if (hash != 0xffffffff) {
					count++;
					h++;

					cells[h] = hash << 1 | 0x01;
					spheres[h] = i << 1 | 0x00;

					m++;
				}
			}

			while (m < 9) {
				h++;
				cells[h] = 0xffffffff;
				spheres[h] = i << 2;
				m++;
			}
		}

		t[threadIdx.x] = count;
		sumReduceCUDA(t, d_temp_ptr);
	};

	__global__ void RadixTabulateCUDA(
		uint32_t* keys,
		uint32_t* radices,
		unsigned int n,
		unsigned int cells_per_group,
		int shift)
	{
		extern __shared__ uint32_t s[];

		int group = threadIdx.x / THREADS_PER_GROUP;
		int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cells_per_group;
		int group_end = group_start + cells_per_group;

		uint32_t k;

		for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i += blockDim.x) {
			s[i] = 0;
		}

		__syncthreads();

		for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end && i < n; i += THREADS_PER_GROUP) {
			k = (keys[i] >> shift & NUM_RADICES - 1) * GROUPS_PER_BLOCK + group;

			for (int j = 0; j < THREADS_PER_GROUP; j++) {
				if (threadIdx.x % THREADS_PER_GROUP == j) {
					s[k]++;
				}
			}
		}

		__syncthreads();

		for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i += blockDim.x) {
			radices[(i / GROUPS_PER_BLOCK * NUM_BLOCKS + blockIdx.x) *
				GROUPS_PER_BLOCK + i % GROUPS_PER_BLOCK] = s[i];
		}
	}

	__global__ void RadixSumCUDA(
		uint32_t* radices,
		uint32_t* radix_sums)
	{

		extern __shared__ uint32_t s[];

		uint32_t total;
		uint32_t left = 0;
		uint32_t* radix = radices + blockIdx.x * NUM_RADICES * GROUPS_PER_BLOCK;

		for (int j = 0; j < NUM_RADICES / NUM_BLOCKS; j++) {
			for (int i = threadIdx.x; i < NUM_BLOCKS * GROUPS_PER_BLOCK; i += blockDim.x) {
				s[i] = radix[i];
			}

			__syncthreads();

			for (int i = threadIdx.x + NUM_BLOCKS * GROUPS_PER_BLOCK; i < PADDED_GROUPS; i += blockDim.x) {
				s[i] = 0;
			}

			__syncthreads();

			if (!threadIdx.x) {
				total = s[PADDED_GROUPS - 1];
			}

			prefixSumCUDA(s, PADDED_GROUPS);
			
			__syncthreads();

			for (int i = threadIdx.x; i < NUM_BLOCKS * GROUPS_PER_BLOCK; i += blockDim.x) {
				radix[i] = s[i];
			}

			if (!threadIdx.x) {
				total += s[PADDED_GROUPS - 1];

				radix_sums[blockIdx.x * NUM_RADICES / NUM_BLOCKS + j] = left;
				total += left;
				left = total;
			}

			radix += NUM_BLOCKS * GROUPS_PER_BLOCK;
		}
	}

	__global__ void RadixReorderCUDA(
		uint32_t* keys_in,
		uint32_t* values_in,
		uint32_t* keys_out,
		uint32_t* values_out,
		uint32_t* radices,
		uint32_t* radix_sums,
		unsigned int n,
		unsigned int cells_per_group,
		int shift)
	{

		extern __shared__ uint32_t s[];

		uint32_t* t = s + NUM_RADICES;

		int group = threadIdx.x / THREADS_PER_GROUP;
		int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cells_per_group;
		int group_end = group_start + cells_per_group;

		uint32_t k;

		for (int i = threadIdx.x; i < NUM_RADICES; i += blockDim.x) {
			s[i] = radix_sums[i];

			if (!((i + 1) % (NUM_RADICES / NUM_BLOCKS))) {
				t[i / (NUM_RADICES / NUM_BLOCKS)] = s[i];
			}
		}

		__syncthreads();

		for (int i = threadIdx.x + NUM_BLOCKS; i < PADDED_BLOCKS; i += blockDim.x) {
			t[i] = 0;
		}

		__syncthreads();

		prefixSumCUDA(t, PADDED_BLOCKS);

		__syncthreads();

		for (int i = threadIdx.x; i < NUM_RADICES; i += blockDim.x) {
			s[i] += t[i / (NUM_RADICES / NUM_BLOCKS)];
		}

		__syncthreads();

		for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i += blockDim.x) {
			t[i] = radices[(i / GROUPS_PER_BLOCK * NUM_BLOCKS + blockIdx.x) *
				GROUPS_PER_BLOCK + i % GROUPS_PER_BLOCK] + s[i / GROUPS_PER_BLOCK];
		}

		__syncthreads();

		for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end && i < n; i += THREADS_PER_GROUP) {
			
			k = (keys_in[i] >> shift & NUM_RADICES - 1) * GROUPS_PER_BLOCK + group;

			for (int j = 0; j < THREADS_PER_GROUP; j++) {
				if (threadIdx.x % THREADS_PER_GROUP == j) {
					keys_out[t[k]] = keys_in[i];
					values_out[t[k]] = values_in[i];
					t[k]++;
				}
			}
		}
	}

	void SpatialPartition::constructCollisionList(
		int m_num_particles, 
		float cell_dim, 
		uint32_t* d_grid_ptr,
		uint32_t* d_sphere_ptr,
		float* d_particle_size_ptr, 
		float4* d_particle_position_ptr, 
		int threads_per_block,
		unsigned int* d_temp_ptr)
	{

		cudaMemset(d_temp_ptr, 0, sizeof(unsigned int));

		constructCollisionListCUDA << <NUM_BLOCKS, threads_per_block, threads_per_block * sizeof(unsigned int) >> > (
			m_num_particles,
			cell_dim, // Max cell size in any dimension
			d_grid_ptr,
			d_sphere_ptr,
			d_particle_size_ptr,
			d_particle_position_ptr,
			d_temp_ptr);
	}

	

	void SpatialPartition::sortCollisionList(
		uint32_t* d_grid_ptr,
		uint32_t* d_sphere_ptr,
		uint32_t* d_grid_temp_ptr,
		uint32_t* d_sphere_temp_ptr,
		uint32_t* d_radices_ptr,
		uint32_t* d_radix_sums_ptr,
		unsigned int n_particles
	)
	{
		unsigned int cells_per_group = (n_particles - 1) / NUM_BLOCKS / GROUPS_PER_BLOCK + 1;

		uint32_t* cells_swap;
		uint32_t* spheres_swap;

		for (int i = 0; i < 32; i += L) {
			
			RadixTabulateCUDA << <NUM_BLOCKS, GROUPS_PER_BLOCK* THREADS_PER_GROUP, GROUPS_PER_BLOCK* NUM_RADICES * sizeof(int) >> > (
				d_grid_ptr,
				d_radices_ptr,
				n_particles * 9,
				cells_per_group,
				i
				);

			RadixSumCUDA << <NUM_BLOCKS, GROUPS_PER_BLOCK* THREADS_PER_GROUP, PADDED_GROUPS * sizeof(int) >> > (
				d_radices_ptr,
				d_radix_sums_ptr
				);

			RadixReorderCUDA << <NUM_BLOCKS, GROUPS_PER_BLOCK* THREADS_PER_GROUP, NUM_RADICES * sizeof(int) + GROUPS_PER_BLOCK * NUM_RADICES * sizeof(int) >> > (
				d_grid_ptr,
				d_sphere_ptr,
				d_grid_temp_ptr,
				d_sphere_temp_ptr,
				d_radices_ptr,
				d_radix_sums_ptr,
				n_particles * 9,
				cells_per_group,
				i
				);

			cells_swap = d_grid_ptr;
			d_grid_ptr = d_grid_temp_ptr;
			d_grid_temp_ptr = cells_swap;

			spheres_swap = d_sphere_ptr;
			d_sphere_ptr = d_sphere_temp_ptr;
			d_sphere_temp_ptr = spheres_swap;
		}
	}
}