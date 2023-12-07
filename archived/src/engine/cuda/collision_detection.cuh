#ifndef _STARDUST_COLLISION_DETECTION_HEADER_
#define _STARDUST_COLLISION_DETECTION_HEADER_

// Internal
#include "../engine.hpp"
#include "cuda_utils.hpp"

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

#define L 8
#define NUM_RADICES 256
#define NUM_BLOCKS 16
#define GROUPS_PER_BLOCK 12
#define THREADS_PER_GROUP 16
#define PADDED_BLOCKS 16
#define PADDED_GROUPS 256

namespace STARDUST {
	
	class SpatialPartition {

	public:
		static void constructCollisionList(
			int m_num_particles,
			float cell_dim,
			uint32_t* d_grid_ptr,
			uint32_t* d_sphere_ptr,
			float* d_particle_size_ptr,
			float4* d_particle_position_ptr,
			int threads_per_block,
			unsigned int* d_temp_ptr);

		static void sortCollisionList(
			uint32_t* d_grid_ptr,
			uint32_t* d_sphere_ptr,
			uint32_t* d_grid_temp_ptr,
			uint32_t* d_sphere_temp_ptr,
			uint32_t* d_radices_ptr,
			uint32_t* d_radix_sums_ptr,
			unsigned int n_particles
		);

		static void tranverseAndResolveCollisionList(
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
		);
	};

	class SpatialPrune {

		static void computeAABB(
			float4* position,
			float* radius,
			float4* lower,
			float4* upper,
			int n_objects
		);
		
		static void projectAABB(
			float4* d_lower_bound_ptr,
			float4* d_upper_bound_ptr,
			float* d_lowerx_ptr,
			float* d_upperx_ptr,
			float* d_lowery_ptr,
			float* d_uppery_ptr,
			float* d_lowerz_ptr,
			float* d_upperz_ptr,
			int n_objects
		);

		// static void clusterPartition(); <-- Spatial partitioning

		static void sortLowerExtents(
			uint32_t* keys_in,
			uint32_t* values_in,
			uint32_t* keys_out,
			uint32_t* values_out,
			uint32_t* radices,
			uint32_t* radix_sums,
			int n
		);

		static void sweepAndPrune(
			float* upperx,
			float* lowerx,
			float* uppery,
			float* lowery,
			float* upperz,
			float* lowerz,
			int* idxx,
			int* idxy,
			int* idxz,
			int* potential_collision,
			int n_objects,
			int padding
		);

	};
}


#endif // _STARDUST_COLLISION_DETECTION_HEADER_

