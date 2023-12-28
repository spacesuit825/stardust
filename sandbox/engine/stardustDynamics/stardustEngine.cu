// C++
#include <iostream>

#include "stardustEngine.hpp"
#include "../stardustUtility/cuda_utils.cuh"
#include "../stardustUtility/util.hpp"

namespace STARDUST {

	void Engine::setupEngine(
		LBVH lbvh_set,
		MPR mpr_set,
		EntityHandler entity_handler_set
	)
	{
		lbvh = lbvh_set;
		mpr = mpr_set;
		entity_handler = entity_handler_set;
	}

	void Engine::allocate() {

		entity_handler.allocate();

		lbvh.allocate(
			entity_handler.getNPrimitives(),
			engine_parameters.max_broad_collisions
		);

		mpr.allocate(
			1,
			entity_handler.getNPrimitives(),
			engine_parameters.max_narrow_collisions
		);
		

		std::cout << "Allocation and Device Transfer success!\n";
	}

	void Engine::reset() {
		
		lbvh.reset();
		mpr.reset();

	}

	__device__ void updatePrimitivePositionsCUDA(
		int tid,
		Hull& hull,
		Entity& entity,
		AABB& aabb,
		float4* d_vertex_ptr,
		float4* d_init_vertex_ptr
	)
	{
		
		float4 rotated_relative_position = multiplyQuaternionByVectorCUDA(entity.quaternion, hull.initial_relative_position);
		
		float4 rotated_upper_extent = multiplyQuaternionByVectorCUDA(entity.quaternion, aabb.init_upper_extent - entity.init_position);
		float4 rotated_lower_extent = multiplyQuaternionByVectorCUDA(entity.quaternion, aabb.init_lower_extent - entity.init_position);

		hull.relative_position = rotated_relative_position;
		hull.position = rotated_relative_position + entity.position;

		aabb.upper_extent = rotated_upper_extent + entity.position;
		aabb.lower_extent = rotated_lower_extent + entity.position;

		if (hull.type == POLYHEDRA) {
			for (int i = hull.vertex_idx; i < hull.vertex_idx + hull.n_vertices; i++) {
				float4 init_vertex = d_init_vertex_ptr[i];
				float4 rotated_vertex = multiplyQuaternionByVectorCUDA(entity.quaternion, init_vertex - entity.init_position);

				d_vertex_ptr[i] = rotated_vertex + entity.position;
			}
		}

		//printf("\nLower extent: %.3f, %.3f, %.3f\n", aabb.upper_extent.x, aabb.upper_extent.y, aabb.upper_extent.z);
	}

	__device__ void updatePrimitiveVelocitiesCUDA(
		int tid,
		Hull& hull,
		Entity& entity
	)
	{
		float4 angular_velocity = entity.angular_velocity;
		float4 relative_position = hull.relative_position;

		float3 linear_velocity = cross(make_float3(angular_velocity), make_float3(relative_position));

		hull.velocity = entity.velocity + make_float4(linear_velocity);
	}


	__global__ void updatePrimitiveDataCUDA(
		int n_primitives,
		Hull* d_hull_ptr,
		Entity* d_entity_ptr,
		AABB* d_aabb_ptr,
		float4* d_vertex_ptr,
		float4* d_init_vertex_ptr
	)
	{
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= n_primitives) return;

		// Load the hull
		Hull hull = d_hull_ptr[tid];
		AABB aabb = d_aabb_ptr[tid];

		int entity_idx = hull.entity_owner;

		
		Entity entity = d_entity_ptr[entity_idx];

		//if (entity_idx == 2) {
		//	printf("entity quat %.3f, %.3f, %.3f\n", entity.quaternion.x, entity.quaternion.y, entity.quaternion.z);
		//}
		
		
		updatePrimitivePositionsCUDA(
			tid,
			hull,
			entity,
			aabb,
			d_vertex_ptr,
			d_init_vertex_ptr
		);
		
		

		updatePrimitiveVelocitiesCUDA(
			tid,
			hull,
			entity
		);

		d_hull_ptr[tid] = hull;
		d_aabb_ptr[tid] = aabb;

	}

	__device__ void atomicAddFloat4(
		int idx,
		float4* array,
		float4& a
	)
	{

		// We could possibly combine these into 64-bit values and perform two atomic adds but I think itll be about the same or worse than this way
		atomicAdd(&array[idx].x, a.x);
		atomicAdd(&array[idx].y, a.y);
		atomicAdd(&array[idx].z, a.z);
	}

	__global__ void resolveCollisionsCUDA(
		int n_collisions,
		CollisionManifold* d_collision_manifold_ptr,
		Hull* d_hull_ptr,
		float4* d_entity_force_ptr,
		float4* d_entity_torque_ptr
	)
	{

		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= n_collisions) return;

		CollisionManifold collision_manifold = d_collision_manifold_ptr[tid];
		
		Hull host_hull = d_hull_ptr[collision_manifold.host_hull_idx];
		Hull phantom_hull = d_hull_ptr[collision_manifold.phantom_hull_idx];

		// if (host_hull.entity_owner == phantom_hull.entity_owner) {
		// 	return;
		// }

		float4 normal_force;
		float4 damping;
		float4 tangent_force;

		float4 relative_position = host_hull.position - phantom_hull.position;
		float4 relative_velocity = host_hull.velocity - phantom_hull.velocity;

		float4 tangent_velocity = relative_velocity - (relative_velocity * (collision_manifold.collision_normal)) * collision_manifold.collision_normal;

		float eff_radius = (host_hull.radius * phantom_hull.radius) / (host_hull.radius + phantom_hull.radius);
		float eff_mass = (host_hull.mass * phantom_hull.mass) / (host_hull.mass + phantom_hull.mass);
		float max_velocity = max(length(host_hull.velocity), length(phantom_hull.velocity));
		float eff_youngs = (host_hull.normal_stiffness * phantom_hull.normal_stiffness) / (host_hull.normal_stiffness + phantom_hull.normal_stiffness);

		float k = (16 / 15) * powf(eff_radius, 0.5f) * eff_youngs * powf(((15 * eff_mass * SQR(max_velocity)) / (16 * powf(eff_radius, 0.5f) * eff_youngs)), 1.0f / 5.0f);
		float c = sqrtf((4 * eff_mass * k) / (1 + SQR(3.1415 / logf(0.5f))));

		normal_force = k * (collision_manifold.collision_normal * collision_manifold.penetration_depth) - c * relative_velocity;

		//normal_force = host_hull.normal_stiffness * (1.5f - collision_manifold.penetration_depth) * collision_manifold.collision_normal;
		//damping = host_hull.damping * dot(collision_manifold.collision_normal, relative_velocity) * collision_manifold.collision_normal;
		
		tangent_force = host_hull.tangential_stiffness * tangent_velocity;

		float4 host_force = (normal_force + tangent_force);
		float4 phantom_force = -(normal_force + tangent_force);

		host_hull.force_application_position = collision_manifold.pointA;
		phantom_hull.force_application_position = collision_manifold.pointB;

		float4 host_application_vector = host_hull.force_application_position - host_hull.relative_position;
		float4 phantom_application_vector = phantom_hull.force_application_position - phantom_hull.relative_position;

		float4 host_torque = make_float4(cross(make_float3(host_application_vector), make_float3(host_force)));
		float4 phantom_torque = make_float4(cross(make_float3(phantom_application_vector), make_float3(phantom_force)));
		

		atomicAddFloat4(host_hull.entity_owner, d_entity_force_ptr, host_force);
		atomicAddFloat4(phantom_hull.entity_owner, d_entity_force_ptr, phantom_force);
		atomicAddFloat4(host_hull.entity_owner, d_entity_torque_ptr, host_torque);
		atomicAddFloat4(phantom_hull.entity_owner, d_entity_torque_ptr, phantom_torque);

	}


	__global__ void collateForcesAndTorquesCUDA(
		int n_entities,
		Hull* d_hull_ptr,
		Entity* d_entity_ptr,
		float4* d_entity_force_ptr,
		float4* d_entity_torque_ptr
	)
	{

		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= n_entities) return;


		Entity entity = d_entity_ptr[tid];

		int primitive_idx = entity.primitive_idx;
		int n_primitives = entity.n_primitives;

		float4 entity_force = make_float4(0.0f);
		float4 entity_torque = make_float4(0.0f);

		// Use warps to parallelise this better
		for (int i = primitive_idx; i < primitive_idx + n_primitives; i++) {

			Hull hull = d_hull_ptr[i];

			entity_force += hull.mass * make_float4(0.0f, 0.0f, -9.81f, 0.0f);
			entity_torque += make_float4(cross(make_float3(hull.relative_position), make_float3(hull.mass * make_float4(0.0f, 0.0f, -9.81f, 0.0f))));

		}

	
		d_entity_force_ptr[tid] += entity_force;
		d_entity_torque_ptr[tid] += entity_torque;

		if (entity.type == MESH) {
			d_entity_force_ptr[tid] = make_float4(0.0f);
			d_entity_torque_ptr[tid] = make_float4(0.0f);
		}
		
	}


	__global__ void integrateForwardCUDA(
		int n_entities,
		float time_step,
		Entity* d_entity_ptr,
		float4* d_entity_force_ptr, 
		float4* d_entity_torque_ptr
	)
	{
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= n_entities) return;

		Entity entity = d_entity_ptr[tid];

		float4 force = d_entity_force_ptr[tid];
		float4 torque = d_entity_torque_ptr[tid];

		// printf("\n Force %.3f, %.3f, %.3f\n", force.x, force.y, force.z);

		entity.linear_momentum += force * time_step;
		entity.angular_momentum += torque * time_step;

		entity.velocity = entity.linear_momentum * (1 / entity.mass);

		entity.position += entity.velocity * time_step;

		float9 inertia_tensor = entity.inertia_tensor;
		float9 inverse_inertia = compute3x3Inverse(inertia_tensor);

		float9 rotation_matrix = quatToRotationCUDA(entity.quaternion);
		float9 rotation_inverse_inertia = computeMatrixMultiplication(rotation_matrix, inverse_inertia);

		float9 inverse_inertia_tensor = computeMatrixMultiplication(rotation_inverse_inertia, computeTranspose(rotation_matrix));

		entity.angular_velocity = computeMatrixVectorMultiplication(inverse_inertia_tensor, entity.angular_momentum);

		float4 angular_displacement = entity.angular_velocity * time_step;

		float angular_displacement_magnitude = length(angular_displacement);
		float angular_velocity_magnitude = length(entity.angular_velocity);

		float4 angular_velocity_normalised;

		// Check against NaN values
		if (angular_displacement_magnitude == 0.0f) {
			angular_velocity_normalised = make_float4(0.0f);
		}
		else {
			angular_velocity_normalised = normalize(entity.angular_velocity);
		}

		float s = cos(((angular_displacement_magnitude) / 2));
		float4 v = angular_velocity_normalised * sin(((angular_displacement_magnitude) / 2));

		// Update quaternion q(t + dt) = (dq x q)
		float4 dq = make_float4(s, v.x, v.y, v.z);
		entity.quaternion = multiplyQuaternionsCUDA(dq, entity.quaternion);

		d_entity_ptr[tid] = entity;

	}

	__global__ void resetEntitiesCUDA(
		int n_entities,
		Entity* d_entity_ptr,
		float4* d_entity_force_ptr,
		float4* d_entity_torque_ptr
	)
	{

		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= n_entities) return;

		d_entity_force_ptr[tid] = make_float4(0.0f);
		d_entity_torque_ptr[tid] = make_float4(0.0f);
		
	}

	__global__ void resetPrimitivesCUDA(
		int n_primitives,
		Hull* d_hull_ptr
	)
	{

		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= n_primitives) return;


		Hull hull = d_hull_ptr[tid];

		hull.force = make_float4(0.0f);
		hull.torque = make_float4(0.0f);

		d_hull_ptr[tid] = hull;
	}
	




	//// HOST SIDE CALLING (in order of kernels above) ////

	void Engine::run() 
	{

		DeviceGeometryData device_data = entity_handler.getDeviceData();
		float time_step = engine_parameters.time_step;

		resetPrimitives(
			entity_handler.getNPrimitives(),
			device_data.d_hull_ptr
		);

		resetEntities(
			entity_handler.getNEntities(),
			device_data.d_entity_ptr,
			device_data.d_entity_force_ptr,
			device_data.d_entity_torque_ptr
		);

		// Primitive Update
		updatePrimitives(
			entity_handler.getNPrimitives(),
			device_data.d_hull_ptr,
			device_data.d_entity_ptr,
			device_data.d_aabb_ptr,
			device_data.d_vertex_ptr,
			device_data.d_init_vertex_ptr
		);

		///// Collision detection ////
		lbvh.execute(
			entity_handler.getNPrimitives(),
			engine_parameters.max_broad_collisions,
			device_data.d_hull_ptr,
			device_data.d_aabb_ptr
		);

		//std::cout << "n collisions: " << lbvh.getCollisionNumber() << "\n";

		mpr.execute(
			lbvh.getCollisionNumber(),
			entity_handler.getNPrimitives(),
			engine_parameters.max_narrow_collisions,
			lbvh.getPotentialCollisionPtr(),
			device_data.d_hull_ptr,
			device_data.d_vertex_ptr
		);

	

		// Resolve collisions
		resolveCollisions(
			mpr.getCollisionNumber(),
			mpr.getCollisionPtr(),
			device_data.d_hull_ptr,
			device_data.d_entity_force_ptr,
			device_data.d_entity_torque_ptr
		);


		// // Collate forces and torques
		collateForcesAndTorques(
			entity_handler.getNEntities(),
			device_data.d_hull_ptr,
			device_data.d_entity_ptr,
			device_data.d_entity_force_ptr,
			device_data.d_entity_torque_ptr
		);

		// Integrate the world
		integrateForward(
			entity_handler.getNEntities(),
			engine_parameters.time_step,
			device_data.d_entity_ptr,
			device_data.d_entity_force_ptr,
			device_data.d_entity_torque_ptr
		);

	}

	void Engine::resetEntities(
		int n_entities,
		Entity* d_entity_ptr,
		float4* d_entity_force_ptr,
		float4* d_entity_torque_ptr
	) 
	{
		unsigned int entity_block_size = 256;
		unsigned int entity_grid_size = (n_entities + entity_block_size - 1) / entity_block_size;

		resetEntitiesCUDA << <entity_grid_size, entity_block_size >> > (
			n_entities,
			d_entity_ptr,
			d_entity_force_ptr,
			d_entity_torque_ptr
			);
	}

	void Engine::resetPrimitives(
		int n_primitives,
		Hull* d_hull_ptr
	)
	{
		unsigned int primitive_block_size = 256;
		unsigned int primitive_grid_size = (n_primitives + primitive_block_size - 1) / primitive_block_size;

		resetPrimitivesCUDA << <primitive_grid_size, primitive_block_size >> > (
			n_primitives,
			d_hull_ptr
			);
	}

	void Engine::updatePrimitives(
		int n_primitives,
		Hull* d_hull_ptr,
		Entity* d_entity_ptr,
		AABB* d_aabb_ptr,
		float4* d_vertex_ptr,
		float4* d_init_vertex_ptr
	)
	{

		unsigned int primitive_block_size = 256;
		unsigned int primitive_grid_size = (n_primitives + primitive_block_size - 1) / primitive_block_size;

		updatePrimitiveDataCUDA << <primitive_grid_size, primitive_block_size >> > (
			n_primitives,
			d_hull_ptr,
			d_entity_ptr,
			d_aabb_ptr,
			d_vertex_ptr,
			d_init_vertex_ptr
			);

	}

	void Engine::resolveCollisions(
		int n_collisions,
		CollisionManifold* d_collision_manifold_ptr,
		Hull* d_hull_ptr,
		float4* d_entity_force_ptr,
		float4* d_entity_torque_ptr
		)
	{
		unsigned int collision_block_size = 256;
		unsigned int collision_grid_size = (n_collisions + collision_block_size - 1) / collision_block_size;

		resolveCollisionsCUDA << <collision_grid_size, collision_block_size >> > (
			n_collisions,
			d_collision_manifold_ptr,
			d_hull_ptr,
			d_entity_force_ptr,
			d_entity_torque_ptr
			);
	}

	void Engine::collateForcesAndTorques(
		int n_entities,
		Hull* d_hull_ptr,
		Entity* d_entity_ptr,
		float4* d_entity_force_ptr,
		float4* d_entity_torque_ptr
	)
	{
		unsigned int entity_block_size = 256;
		unsigned int entity_grid_size = (n_entities + entity_block_size - 1) / entity_block_size;

		collateForcesAndTorquesCUDA << <entity_grid_size, entity_block_size >> > (
			n_entities,
			d_hull_ptr,
			d_entity_ptr,
			d_entity_force_ptr,
			d_entity_torque_ptr
			);
	}

	void Engine::integrateForward(
		int n_entities,
		float time_step,
		Entity* d_entity_ptr, 
		float4* d_entity_force_ptr, 
		float4* d_entity_torque_ptr
	)
	{

		//printf("n entities %d\n", n_entities);
		unsigned int entity_block_size = 256;
		unsigned int entity_grid_size = (n_entities + entity_block_size - 1) / entity_block_size;

		integrateForwardCUDA << <entity_grid_size, entity_block_size >> > (
			n_entities,
			time_step,
			d_entity_ptr,
			d_entity_force_ptr, 
			d_entity_torque_ptr
			);
	}



}