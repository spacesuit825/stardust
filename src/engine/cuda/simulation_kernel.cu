// Internal
#include "../engine.hpp"
#include "cuda_utils.hpp"
#include "collision_detection.cuh"
#include "physics_update.cuh"

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

	void DEMEngine::step(Scalar timestep) {

		// Steps
		// 1. Update sphere positions and velocities (rotation and translation) based on new particle position
		// 2. Compute relative positions of spheres to entity COM
		// 3. Compute collisions
		// 4. Compute particle force and torques from spheres
		// 5. Compute momentum (linear and angular) on particle
		// 6. Advect particle and compute quaternion

		float cell_dim = 0.2 * 2;

		int threads_per_block = 128;
		unsigned int particle_size = (m_num_particles - 1) / threads_per_block + 1;

		// PARTICLE UPDATES AND RELATIVE POSITIONS


		// COLLISION DETECTION AND RESPONSE //
		
		constructCollisionList(
			m_num_particles, 
			cell_dim, 
			d_grid_ptr, 
			d_sphere_ptr, 
			d_particle_size_ptr, 
			d_particle_position_ptr, 
			threads_per_block,
			d_temp_ptr
		);

		unsigned int count;
		cudaMemcpy(&count, d_temp_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		std::cout << "Cells Occupied " << count << "\n";

		sortCollisionList(
			d_grid_ptr,
			d_sphere_ptr,
			d_grid_temp_ptr,
			d_sphere_temp_ptr,
			d_radices_ptr,
			d_radix_sums_ptr,
			m_num_particles
		);

		tranverseAndResolveCollisionList(
			d_grid_ptr,
			d_sphere_ptr,
			d_particle_position_ptr,
			d_particle_velocity_ptr,
			d_particle_forces_ptr,
			d_particle_mass_ptr,
			d_particle_size_ptr,
			d_particle_to_rigid_idx_ptr,
			m_num_particles,
			d_temp_ptr,
			threads_per_block
		);

		float4 force0;
		float4 force1;
		CUDA_ERR_CHECK(cudaMemcpy(&force0, d_particle_forces_ptr, sizeof(float4), cudaMemcpyDeviceToHost));
		CUDA_ERR_CHECK(cudaMemcpy(&force1, d_particle_forces_ptr + 1, sizeof(float4), cudaMemcpyDeviceToHost));

		printf("Force on Particle 0: %.3f, %.3f, %.3f\n", force0.x, force0.y, force0.z);
		printf("Force on Particle 1: %.3f, %.3f, %.3f\n", force1.x, force1.y, force1.z);

		// PARTICLE FORCE COMPUTATION AND POSITION/ORIENTATION UPDATE

	}
}