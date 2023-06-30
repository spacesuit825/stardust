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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvfunctional>


namespace STARDUST {

	__global__ void constructCellsCUDA(
		unsigned int n_spheres,
		float cell_dim, // Max cell size in any dimension
		int* cells,
		int* spheres,
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

			int hash = 0;
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

	__global__ void collideCellsCUDA(
		int* cells,
		int* spheres,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float* d_particle_size_ptr,
		unsigned int n_particles,
		unsigned int n_cells,
		unsigned int cells_per_thread,
		unsigned int* collision_count,
		unsigned int* test_count
	)
	{
		extern __shared__ unsigned int t[];

		int thread_start = (blockIdx.x * blockDim.x + threadIdx.x) * cells_per_thread;
		int thread_end = thread_start + cells_per_thread;

		int start = -1;
		int i = thread_start;

		int last = 0xffffffff;
		int home;
		int phantom;

		unsigned int h;
		unsigned int p;
		unsigned int collisions = 0;
		unsigned int tests = 0;

		float dh;
		float dp;
		float dx;
		float d;

		while (1) {
			if (i >= n_cells || cells[i] >> 1 != last) {
				if (start + 1 && h >= 1 && h + p >= 2) {
					for (int j = start; j < start + h; j++) {
						home = spheres[j] >> 1;
						dh = d_particle_size_ptr[home];

						for (int k = j + 1; k < i; k++) {
							tests++;
							phantom = spheres[k] >> 1;
							dp = d_particle_size_ptr[phantom] + dh;
							d = 0;

							float4 p_home = d_particle_position_ptr[home];
							float4 p_phantom = d_particle_position_ptr[phantom];

							float p_h[3] = { p_home.x, p_home.y, p_home.z };
							float p_p[3] = { p_phantom.x, p_phantom.y, p_phantom.z };

							for (int l = 0; l < 3; l++) {
								dx = p_p[l] - p_h[l];

								d += dx * dx;
							}

							if (d < dp * dp) {
								collisions++;
							}
						}
					}
				}

				if (i > thread_end || i >= m) {
					break;
				}

				if (i != thread_start || !blockIdx.x && !threadIdx.x) {
					h = 0;
					p = 0;
					start = i;
				}

				last = cells[i] >> 1;
			}

			if (start + 1) {
				if (spheres[i] & 0x01) {
					h++;
				}
				else {
					p++;
				}
			}

			i++;
		}

		t[threadIdx.x] = collisions;
		sumReduceCUDA(t, collision_count);

		__syncthreads();

		t[threadIdx.x] = tests;
		sumReduceCUDA(t, test_count);
	}

	void constructCells(
		int m_num_particles, 
		float cell_dim, 
		int* d_grid_ptr, 
		int* d_sphere_ptr, 
		float* d_particle_size_ptr, 
		float4* d_particle_position_ptr, 
		int threads_per_block,
		unsigned int* d_temp_ptr)
	{

		cudaMemset(d_temp_ptr, 0, sizeof(unsigned int));

		constructCellsCUDA << <NUM_BLOCKS, threads_per_block, threads_per_block * sizeof(unsigned int) >> > (
			m_num_particles,
			cell_dim, // Max cell size in any dimension
			d_grid_ptr,
			d_sphere_ptr,
			d_particle_size_ptr,
			d_particle_position_ptr,
			d_temp_ptr);
	}

	__global__ void collideCellsCUDA(
		int* d_grid_ptr,
		int* d_sphere_ptr,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float* d_particle_size_ptr,
		unsigned int n_particles,
		unsigned int* d_temp_ptr,
		int threads_per_block
	) 
	{
		int n_cells = d_temp_ptr[0];

		unsigned int cells_per_thread = (n_cells - 1) / NUM_BLOCKS /
			threads_per_block + 1;
		unsigned int collision_count;

		cudaMemset(d_temp_ptr, 0, 2 * sizeof(unsigned int));

		collideCellsCUDA << <NUM_BLOCKS, threads_per_block, threads_per_block * sizeof(unsigned int) >> > (
			cells,
			spheres,
			d_particle_position_ptr,
			d_particle_velocity_ptr,
			d_particle_size_ptr,
			n_particles,
			n_cells,
			cells_per_thread,
			temp,
			temp + 1
			);
	}

	__global__ void sortCellsCUDA(
		int* d_grid_ptr,
		int* d_sphere_ptr,
		int* d_grid_temp_ptr,
		int* d_sphere_temp_ptr,
		int* d_radices_ptr,
		int* d_radix_sums_ptr,
		unsigned int n_particles
	)
	{
		unsigned int cells_per_group = (n_particles * 9 - 1) / NUM_BLOCKS / GROUPS_PER_BLOCK + 1;

		int* cells_swap;
		int* spheres_swap;

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