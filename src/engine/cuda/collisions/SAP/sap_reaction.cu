// Internal
#include "../../cuda_utils.hpp"
#include "../../cuda_helpers.cuh"
#include "sap_collision.cuh"

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

	__device__ void computeParticleToParticleCUDA(
		int home_idx,
		int phantom_idx,
		float normal_stiffness,
		float tangential_stiffness,
		float damping,
		float4* d_particle_forces_ptr,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float* d_particle_size_ptr
	)
	{
		float4 normal_force = make_float4(0.0, 0.0, 0.0, 0.0);
		float4 damping_force = make_float4(0.0, 0.0, 0.0, 0.0);
		float4 tangent_force = make_float4(0.0, 0.0, 0.0, 0.0);

		float4 normalised_position;

		float4 relative_position = d_particle_position_ptr[home_idx] - d_particle_position_ptr[phantom_idx];
		float4 relative_velocity = d_particle_velocity_ptr[home_idx] - d_particle_velocity_ptr[phantom_idx];
		float distance = length(relative_position);

		float separation = d_particle_size_ptr[home_idx] + d_particle_size_ptr[phantom_idx];

		if (distance <= separation) {

			if (distance == 0.0) {
				normalised_position = make_float4(0.0, 0.0, 0.0, 0.0);
			}
			else {
				normalised_position = relative_position / distance;
			}

			float4 tangent_velocity = relative_velocity - (relative_velocity * (normalised_position)) * (normalised_position);

			normal_force = -normal_stiffness * (separation - distance) * (normalised_position);
			damping_force = damping * relative_velocity;
			tangent_force = tangential_stiffness * tangent_velocity;
		}

		float4 signed_velocity = convertVectorToSigns(relative_velocity);

		float4 total_force = normal_force + damping_force + tangent_force;

		/*if (home_idx == 0) {
			printf("total force: %.3f, %.3f, %.3f\n", total_force.x, total_force.y, total_force.z);
		}*/

		d_particle_forces_ptr[home_idx] += signed_velocity * total_force;
		d_particle_forces_ptr[phantom_idx] += -signed_velocity * total_force;
	}

	__global__ void reactCollisionsCUDA(
		int n_objects,
		int max_collisions,
		int* d_potential_collision_ptr,
		float4* d_particle_forces_ptr,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float* d_particle_size_ptr,
		int* d_particle_to_rigid_idx_ptr)
	{

		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= n_objects) {
			return;
		}

		int home_idx = tid;

		int collisions_idx = home_idx * max_collisions;

		// TODO: Add these to sphere arrays !!
		float normal_stiffness = 2e6;
		float tangential_stiffness = 20.0;
		float damping = 2000.0;

		for (int i = 0; i < max_collisions; i++) {

			int phantom_idx = d_potential_collision_ptr[collisions_idx + i];

			if (phantom_idx < 0) {
				continue;
			}

			if (d_particle_to_rigid_idx_ptr[home_idx] == d_particle_to_rigid_idx_ptr[phantom_idx]) {
				continue;
			}

			// TODO: Type analysis
			/*if (type == 0) {

			}
			else if (type == 1) {

			}*/
			computeParticleToParticleCUDA(
				home_idx,
				phantom_idx,
				normal_stiffness,
				tangential_stiffness,
				damping,
				d_particle_forces_ptr,
				d_particle_position_ptr,
				d_particle_velocity_ptr,
				d_particle_size_ptr
				);
		}

		/*if (tid == 0) {
			printf("Forces: %.3f, %.3f, %.3f, %.3f\n", d_particle_forces_ptr[home_idx]);
		}*/

	}

	void SAPCollision::reactCollisions(
		int n_objects,
		int max_collisions,
		float4* d_particle_forces_ptr,
		float4* d_particle_position_ptr,
		float4* d_particle_velocity_ptr,
		float* d_particle_size_ptr,
		int* d_particle_to_rigid_idx_ptr
	) 
	{

		int threadsPerBlock = 256;
		int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

		reactCollisionsCUDA << <numBlocks, threadsPerBlock >> > (
			n_objects,
			max_collisions,
			d_potential_collision_ptr,
			d_particle_forces_ptr,
			d_particle_position_ptr,
			d_particle_velocity_ptr,
			d_particle_size_ptr,
			d_particle_to_rigid_idx_ptr
			);
	}

}