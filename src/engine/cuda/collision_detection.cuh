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
	
	void constructCollisionList(
		int m_num_particles,
		float cell_dim,
		uint32_t* d_grid_ptr,
		uint32_t* d_sphere_ptr,
		float* d_particle_size_ptr,
		float4* d_particle_position_ptr,
		int threads_per_block,
		unsigned int* d_temp_ptr);

	void sortCollisionList(
		uint32_t* d_grid_ptr,
		uint32_t* d_sphere_ptr,
		uint32_t* d_grid_temp_ptr,
		uint32_t* d_sphere_temp_ptr,
		uint32_t* d_radices_ptr,
		uint32_t* d_radix_sums_ptr,
		unsigned int n_particles
	);

	void tranverseCollisionList(
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
}


#endif // _STARDUST_COLLISION_DETECTION_HEADER_

