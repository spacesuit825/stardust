// Internal
#include "../engine.hpp"
#include "cuda_utils.hpp"
#include "./cuda/collisions/SAP/sap_collision.cuh"
#include "physics_update.cuh"

// C++
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <thread>

// CUDA
#include <vector_types.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvfunctional>

using namespace std::chrono_literals; // ns, us, ms, s, h, etc.
using std::chrono::system_clock;


namespace STARDUST {

	void DEMEngine::cleanBuffers() {
		CUDA_ERR_CHECK(cudaDeviceSynchronize());

		CUDA_ERR_CHECK(cudaMemset(
			entity_handler.d_particle_forces_ptr,
			0,
			entity_handler.getNumberOfSpheres() * sizeof(float4))
		);

		CUDA_ERR_CHECK(cudaMemset(
			entity_handler.d_rigid_body_forces_ptr,
			0,
			entity_handler.getNumberOfParticles() * sizeof(float4))
		);

		CUDA_ERR_CHECK(cudaMemset(
			entity_handler.d_rigid_body_torques_ptr,
			0,
			entity_handler.getNumberOfParticles() * sizeof(float4))
		);
	}

	void DEMEngine::update(float dt) {

		// Steps
		// 1. Update sphere positions and velocities (rotation and translation) based on new particle position
		// 2. Compute relative positions of spheres to entity COM
		// 3. Compute collisions
		// 4. Compute particle force and torques from spheres
		// 5. Compute momentum (linear and angular) on particle
		// 6. Advect particle and compute quaternion

		float cell_dim = 0.2 * 2;

		int threads_per_block = 128;
		unsigned int particle_size = (entity_handler.getNumberOfSpheres() - 1) / threads_per_block + 1;

		

		// PARTICLE UPDATES AND RELATIVE POSITIONS
		updateSphereData(
			entity_handler.getNumberOfSpheres(),
			entity_handler.getNumberOfParticles(),
			entity_handler.d_particle_position_ptr,
			entity_handler.d_particle_velocity_ptr,
			entity_handler.d_particle_init_relative_position_ptr,
			entity_handler.d_particle_relative_position_ptr,
			entity_handler.d_particle_to_rigid_idx_ptr,
			entity_handler.d_rigid_body_position_ptr,
			entity_handler.d_rigid_body_velocity_ptr,
			entity_handler.d_rigid_body_quaternion_ptr,
			entity_handler.d_rigid_body_angular_velocity_ptr,
			particle_size
		);

		cleanBuffers();

		std::chrono::time_point<std::chrono::system_clock> start;
		std::chrono::duration<double> duration;

		double time;
		start = std::chrono::system_clock::now();

		collision_handler.processCollisions(
			entity_handler.d_particle_position_ptr,
			entity_handler.d_particle_size_ptr,
			entity_handler.getNumberOfSpheres(),
			10);

		collision_handler.reactCollisions(
			entity_handler.getNumberOfSpheres(),
			10,
			entity_handler.d_particle_forces_ptr,
			entity_handler.d_particle_position_ptr,
			entity_handler.d_particle_velocity_ptr,
			entity_handler.d_particle_size_ptr,
			entity_handler.d_particle_to_rigid_idx_ptr);

		// PARTICLE FORCE COMPUTATION AND POSITION/ORIENTATION UPDATE

		duration = std::chrono::system_clock::now() - start;

		time = duration.count();

		std::cout << "Analysis completed in: " << time << "s\n";
		

		computeForcesAndTorquesOnParticles(
			entity_handler.getNumberOfSpheres(),
			entity_handler.getNumberOfParticles(),
			entity_handler.d_rigid_body_forces_ptr,
			entity_handler.d_rigid_body_torques_ptr,
			entity_handler.d_rigid_body_mass_ptr,
			entity_handler.d_entity_start_ptr,
			entity_handler.d_entity_length_ptr,
			entity_handler.d_particle_relative_position_ptr,
			entity_handler.d_particle_forces_ptr,
			entity_handler.d_particle_to_rigid_idx_ptr
		);

		

		advectParticles(
			entity_handler.getNumberOfSpheres(),
			entity_handler.getNumberOfParticles(),
			dt,
			entity_handler.d_rigid_body_position_ptr,
			entity_handler.d_rigid_body_velocity_ptr,
			entity_handler.d_rigid_body_angular_velocity_ptr,
			entity_handler.d_rigid_body_forces_ptr,
			entity_handler.d_rigid_body_torques_ptr,
			entity_handler.d_rigid_body_quaternion_ptr,
			entity_handler.d_rigid_body_mass_ptr,
			entity_handler.d_rigid_body_linear_momentum_ptr,
			entity_handler.d_rigid_body_angular_momentum_ptr,
			entity_handler.d_rigid_body_inertia_tensor_ptr
		);

		
	}
}