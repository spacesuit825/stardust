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

namespace STARDUST {

	void updateSphereData(
		int n_spheres,
		int n_particles,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float4* d_particle_init_relative_position_ptr,
		float4* d_particle_relative_position_ptr,
		int* d_particle_to_rigid_idx_ptr,
		float4* d_rigid_body_position_ptr,
		float4* d_rigid_body_velocity_ptr,
		float4* d_rigid_body_quaternion_ptr,
		float4* d_rigid_body_angular_velocity_ptr,
		int particle_size);

	void computeForcesAndTorquesOnParticles(
		int n_particles,
		int n_entities,
		float4* d_rigid_body_forces_ptr,
		float4* d_rigid_body_torques_ptr,
		float* d_rigid_body_mass_ptr,
		int* d_entity_start_ptr,
		int* d_entity_length_ptr,
		float4* d_particle_relative_position_ptr,
		float4* d_particle_forces_ptr,
		int* d_particle_to_rigid_idx_ptr
	);

	void advectParticles(
		int m_num_particles,
		int m_num_entities,
		float timestep,
		float4* d_rigid_body_position_ptr,
		float4* d_rigid_body_velocity_ptr,
		float4* d_rigid_body_angular_velocity_ptr,
		float4* d_rigid_body_forces_ptr,
		float4* d_rigid_body_torques_ptr,
		float4* d_rigid_body_quaternion_ptr,
		float* d_rigid_body_mass_ptr,
		float4* d_rigid_body_linear_momentum_ptr,
		float4* d_rigid_body_angular_momentum_ptr,
		float9* d_rigid_body_inertia_tensor_ptr
	);
}