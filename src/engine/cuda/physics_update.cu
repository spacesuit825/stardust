#ifndef CUDACC
#define CUDACC
#endif

// Internal
#include "../engine.hpp"
#include "cuda_utils.hpp"
#include "physics_update.cuh"
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

	__device__ void updateRelativePositionsCUDA(
		unsigned int idx,
		int rigid_body_idx,
		float4* d_particle_relative_position_ptr,
		float4* d_particle_init_relative_position_ptr,
		float4* d_rigid_body_position_ptr,
		float4* d_rigid_body_quaternion_ptr
	)
	{
		float4 init_relative_position = d_particle_init_relative_position_ptr[idx];
		float4 rigid_body_quaternion = d_rigid_body_quaternion_ptr[rigid_body_idx];

		float4 rotated_relative_position = multiplyQuaternionsCUDA(
												rigid_body_quaternion,
												init_relative_position);
		
		d_particle_relative_position_ptr[idx] = rotated_relative_position;
	}

	__device__ void updateSpherePositionsCUDA(
		unsigned int idx,
		int rigid_body_idx,
		float4* d_particle_position_ptr,
		float4* d_particle_relative_position_ptr,
		float4* d_rigid_body_position_ptr
	)
	{
		d_particle_position_ptr[idx] = d_particle_relative_position_ptr[idx] + d_rigid_body_position_ptr[rigid_body_idx];
	}

	__device__ void updateSphereVelocitiesCUDA(
		unsigned int idx,
		int rigid_body_idx,
		float4* d_particle_velocity_ptr,
		float4* d_particle_relative_position_ptr,
		float4* d_rigid_body_velocity_ptr,
		float4* d_rigid_body_angular_velocity_ptr
	)
	{
		float4 angular_velocity = d_rigid_body_angular_velocity_ptr[rigid_body_idx];
		float4 relative_position = d_particle_relative_position_ptr[idx];

		// Cross product only supported for vec3, make conversion
		float3 angular_to_linear_velocity = cross(make_float3(angular_velocity.x, angular_velocity.y, angular_velocity.z), make_float3(relative_position.x, relative_position.y, relative_position.z));
		
		// Convert back to supported vec4
		d_particle_velocity_ptr[idx] = d_rigid_body_velocity_ptr[rigid_body_idx] + make_float4(angular_to_linear_velocity.x, angular_to_linear_velocity.y, angular_to_linear_velocity.z, 0.0f);
	}

	__global__ void updateSphereDataCUDA(
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
		float4* d_rigid_body_angular_velocity_ptr
	)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

		// Get entity owner
		int rigid_body_idx = d_particle_to_rigid_idx_ptr[idx];

		// Update relative positions
		updateRelativePositionsCUDA(
			idx,
			rigid_body_idx,
			d_particle_relative_position_ptr,
			d_particle_init_relative_position_ptr,
			d_rigid_body_position_ptr,
			d_rigid_body_quaternion_ptr
		);

		// Update sphere positions
		updateSpherePositionsCUDA(
			idx,
			rigid_body_idx,
			d_particle_position_ptr,
			d_particle_relative_position_ptr,
			d_rigid_body_position_ptr
		);

		// Update sphere velocities
		updateSphereVelocitiesCUDA(
			idx,
			rigid_body_idx,
			d_particle_velocity_ptr,
			d_particle_relative_position_ptr,
			d_rigid_body_velocity_ptr,
			d_rigid_body_angular_velocity_ptr
		);
	}

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
		int particle_size
	)
	{
		unsigned int particle_block_size = 64;
		unsigned int particle_grid_size = (n_spheres + particle_block_size - 1) / particle_block_size;

		updateSphereDataCUDA << < particle_grid_size, particle_block_size >> > (
			n_spheres,
			n_particles,
			d_particle_position_ptr,
			d_particle_velocity_ptr,
			d_particle_init_relative_position_ptr,
			d_particle_relative_position_ptr,
			d_particle_to_rigid_idx_ptr,
			d_rigid_body_position_ptr,
			d_rigid_body_velocity_ptr,
			d_rigid_body_quaternion_ptr,
			d_rigid_body_angular_velocity_ptr
			);
	}

	///////////////////////////////////// COMPUTE FORCES AND TORQUES ON THE RIGID BODY /////////////////////////////////////////////

	__global__ void computeForcesAndTorquesCUDA(
		int n_particles,
		int n_entities,
		float4* d_rigid_body_forces_ptr,
		float4* d_rigid_body_torques_ptr,
		int* d_entity_start_ptr,
		int* d_entity_length_ptr,
		float4* d_particle_relative_position_ptr,
		float4* d_particle_forces_ptr,
		int* d_particle_to_rigid_idx_ptr
	)
	{
		unsigned int rigid_body_idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (rigid_body_idx >= n_entities) {
			return;
		}

		int start = d_entity_start_ptr[rigid_body_idx];
		int length = d_entity_length_ptr[rigid_body_idx];


		float4 force_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 torque_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		// Start is the first index of the entity in the particle array
		for (int i = start; i < start + length; i++) {

			float4 force = d_particle_forces_ptr[i];
			float4 relative_position = d_particle_relative_position_ptr[i];
			
			force_sum += force;

			float3 torque = cross(make_float3(relative_position.x, relative_position.y, relative_position.z), make_float3(force.x, force.y, force.z));
			torque_sum += make_float4(torque.x, torque.y, torque.z, 0.0f);
		}

		d_rigid_body_forces_ptr[rigid_body_idx] = force_sum;
		d_rigid_body_torques_ptr[rigid_body_idx] = torque_sum;
	}

	void computeForcesAndTorquesOnParticles(
		int n_particles,
		int n_entities,
		float4* d_rigid_body_forces_ptr,
		float4* d_rigid_body_torques_ptr,
		int* d_entity_start_ptr,
		int* d_entity_length_ptr,
		float4* d_particle_relative_position_ptr,
		float4* d_particle_forces_ptr,
		int* d_particle_to_rigid_idx_ptr
	)
	{
		unsigned int entity_block_size = 64;
		unsigned int entity_grid_size = (n_entities + entity_block_size - 1) / entity_block_size;

		// Launch a kernel with a thread per entity
		computeForcesAndTorquesCUDA << <entity_grid_size, entity_block_size >> > (
			n_particles,
			n_entities,
			d_rigid_body_forces_ptr,
			d_rigid_body_torques_ptr,
		    d_entity_start_ptr,
			d_entity_length_ptr,
			d_particle_relative_position_ptr,
			d_particle_forces_ptr,
			d_particle_to_rigid_idx_ptr
			);
	}

	///////////////////////////////////////////////////// COMPUTE LINEAR AND ANGULAR MOMENTUM ON RIGID BODY ///////////////////////////////////////////////


}