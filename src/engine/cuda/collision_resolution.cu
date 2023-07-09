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

// Checks potential collisions found through collision_detection.cu
// Resolves the collision via the nominated method
// Applies the necessary forces to the particles in question

namespace STARDUST {

	__device__ void resolveCollisionsCUDA(
		// Add Engine flags to control resolution method
		int cell_idx,
		int start,
		unsigned int h,
		unsigned int p,
		uint32_t* spheres,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float4* d_particle_force_ptr,
		float* d_particle_mass_ptr,
		float* d_particle_size_ptr,
		int* d_particle_to_rigid_idx_ptr
	)
	{
		int home; // Home particle index
		int phantom; // Phantom particle index

		float particle_diameter = 0.2;

		// TODO: Add these to sphere arrays !!
		float normal_stiffness = 2e7;
		float tangential_stiffness = 2.0;
		float damping = 2.0;

		// Use Basic Elastic Resolution for now
		for (int j = start; j < start + h; j++) {

			float4 normal_force = make_float4(0.0, 0.0, 0.0, 0.0);
			float4 damping_force = make_float4(0.0, 0.0, 0.0, 0.0);
			float4 tangent_force = make_float4(0.0, 0.0, 0.0, 0.0);

			home = spheres[j] >> 1;

			for (int k = j + 1; k < cell_idx; k++) {

				phantom = spheres[k] >> 1;

				// Check if phantom sphere is a member of the same particle
				if (d_particle_to_rigid_idx_ptr[home] == d_particle_to_rigid_idx_ptr[phantom]) {
					continue;
				}
				
				//TODO: Compute secant normal stiffness (for now assume the values are constant)

				float4 relative_position = d_particle_position_ptr[home] - d_particle_position_ptr[phantom];
				float4 relative_velocity = d_particle_velocity_ptr[home] - d_particle_velocity_ptr[phantom];
				float distance = length(relative_position);
				
				if (distance <= particle_diameter) {

					float4 normalised_position = relative_position / distance;

					float4 tangent_velocity = relative_velocity - (relative_velocity * (normalised_position)) * (normalised_position);

					normal_force = -normal_stiffness * (particle_diameter - distance) * (normalised_position);
					damping_force = damping * relative_velocity;
					tangent_force = tangential_stiffness * tangent_velocity;	
				}

				float4 signed_velocity = convertVectorToSigns(relative_velocity);
				
				float4 total_force = normal_force + damping_force + tangent_force;

				d_particle_force_ptr[home] += -signed_velocity * total_force;
				d_particle_force_ptr[phantom] = signed_velocity * total_force;
			}
		}
	}

	__global__ void tranverseCollisionListCUDA(
		uint32_t* cells,
		uint32_t* spheres,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float4* d_particle_force_ptr,
		float* d_particle_mass_ptr,
		float* d_particle_size_ptr,
		int* d_particle_to_rigid_idx_ptr,
		unsigned int n_particles,
		unsigned int* collision_count,
		unsigned int* test_count,
		int threads_per_block
	)
	{
		extern __shared__ unsigned int t[];

		int n_cells = collision_count[0];

		collision_count[0] = 0;
		test_count[0] = 0;

		unsigned int cells_per_thread = (n_cells - 1) / NUM_BLOCKS /
			threads_per_block + 1;

		int thread_start = (blockIdx.x * blockDim.x + threadIdx.x) * cells_per_thread;
		int thread_end = thread_start + cells_per_thread;

		int start = -1;
		int i = thread_start;

		uint32_t last = 0xffffffff;
		uint32_t home;
		uint32_t phantom;

		unsigned int h;
		unsigned int p;
		unsigned int collisions = 0;
		unsigned int tests = 0;

		float dh;
		float dp;
		float dx;
		float d;

		while (1) {
			
			// Check for cell ID changes (indicates potential collision)
			if (i >= n_cells || cells[i] >> 1 != last) {
				// Check surroundings for home volume and at minimum one other
				if (start + 1 && h >= 1 && h + p >= 2) {
	
					resolveCollisionsCUDA(
						i,
						start,
						h,
						p,
						spheres,
						d_particle_position_ptr,
						d_particle_velocity_ptr,
						d_particle_force_ptr,
						d_particle_mass_ptr,
						d_particle_size_ptr,
						d_particle_to_rigid_idx_ptr);

				}

				if (i > thread_end || i >= n_cells) {
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

	void tranverseAndResolveCollisionList(
		uint32_t* d_grid_ptr,
		uint32_t* d_sphere_ptr,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float4* d_particle_force_ptr,
		float* d_particle_mass_ptr,
		float* d_particle_size_ptr,
		int* d_particle_to_rigid_idx_ptr,
		unsigned int n_particles,
		unsigned int* d_temp_ptr,
		int threads_per_block
	)
	{
		unsigned int collision_count;

		tranverseCollisionListCUDA << <NUM_BLOCKS, threads_per_block, threads_per_block * sizeof(unsigned int) >> > (
			d_grid_ptr,
			d_sphere_ptr,
			d_particle_position_ptr,
			d_particle_velocity_ptr,
			d_particle_force_ptr,
			d_particle_mass_ptr,
			d_particle_size_ptr,
			d_particle_to_rigid_idx_ptr,
			n_particles,
			d_temp_ptr,
			d_temp_ptr + 1,
			threads_per_block
			);
	}

}

//for (int j = start; j < start + h; j++) {
//
//	home = spheres[j] >> 1;
//	// Record the home particle id
//
//	dh = d_particle_size_ptr[home];
//
//	for (int k = j + 1; k < i; k++) {
//		tests++;
//		phantom = spheres[k] >> 1;
//
//
//		dp = d_particle_size_ptr[phantom] + dh;
//		d = 0;
//
//		float4 p_home = d_particle_position_ptr[home];
//		float4 p_phantom = d_particle_position_ptr[phantom];
//
//		float4 distance = p_home - p_phantom;
//
//		float d = length(distance);
//
//		printf("%.3f, %.3f\n", d, dp);
//
//		if (abs(d) < dp) {
//			collisions++;
//		}
//	}
//}