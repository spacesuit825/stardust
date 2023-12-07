// Internal
#include "../../cuda_utils.hpp"
#include "sap_collision.cuh"
#include "../../collision_detection.cuh"

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

namespace STARDUST {

	void SAPCollision::initiatePointers() {

		d_idx_ptr = nullptr;

		d_lower_bound_ptr = nullptr;
		d_upper_bound_ptr = nullptr;

		d_lower_extent_x_ptr = nullptr;
		d_upper_extent_x_ptr = nullptr;

		d_lower_extent_y_ptr = nullptr;
		d_upper_extent_y_ptr = nullptr;

		d_lower_extent_z_ptr = nullptr;
		d_upper_extent_z_ptr = nullptr;

		d_potential_collision_ptr = nullptr;

		d_temp_key_ptr = nullptr;
		d_temp_value_ptr = nullptr;

		d_radix_ptr = nullptr;
		d_radix_sum_ptr = nullptr;
	}

	void SAPCollision::prepareData() {

	}

	void SAPCollision::allocateCUDA(int n_spheres, int max_collisions) {

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_idx_ptr,
			n_spheres * sizeof(int)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_lower_bound_ptr,
			n_spheres * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_upper_bound_ptr,
			n_spheres * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_lower_extent_x_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_upper_extent_x_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_lower_extent_y_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_upper_extent_y_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_lower_extent_z_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_upper_extent_z_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_potential_collision_ptr,
			(max_collisions * n_spheres + max_collisions) * sizeof(int)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_temp_key_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_temp_value_ptr,
			n_spheres * sizeof(int)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_radix_ptr,
			NUM_BLOCKS * NUM_RADICES * GROUPS_PER_BLOCK * sizeof(int)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_radix_sum_ptr,
			NUM_RADICES * sizeof(int)
		));

	}

	void SAPCollision::transferDataToDevice() {

	}

	// Process Collisions //

	// Clear and initiate some tracking arrays
	__global__ void initIdxCUDA(
		int n_objects,
		int max_collisions,
		int* d_idx_ptr,
		int* d_potential_collision_ptr
	)
	{
		
		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= n_objects) {
			return;
		}

		d_idx_ptr[tid] = tid;

		for (int i = 0; i < max_collisions; i++) {
			d_potential_collision_ptr[tid + i] = -1;
		}

	}

	void SAPCollision::initIdx(
		int n_objects,
		int max_collisions)
	{

		int threadsPerBlock = 256;
		int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

		initIdxCUDA << <numBlocks, threadsPerBlock >> > (
			n_objects,
			max_collisions,
			d_idx_ptr,
			d_potential_collision_ptr
			);

	}




	// Compute object AABBs
	__global__ void computeAABBCUDA(
		int n_objects,
		float4* position,
		float* radius,
		float4* lower,
		float4* upper
	)
	{
		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= n_objects) {
			return;
		}

		float4 pos = position[tid];
		float r = radius[tid];

		upper[tid] = make_float4(pos.x + r, pos.y + r, pos.z + r, 0.0f);
		lower[tid] = make_float4(pos.x - r, pos.y - r, pos.z - r, 0.0f);

	}

	void SAPCollision::computeAABB(
		float4* d_position_ptr, 
		float* d_radius_ptr,
		int n_objects) 
	{

		int threadsPerBlock = 256;
		int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

		computeAABBCUDA << < numBlocks, threadsPerBlock >> > (
			n_objects,
			d_position_ptr,
			d_radius_ptr,
			d_lower_bound_ptr,
			d_upper_bound_ptr
			);

	}




	__global__ void projectAABBCUDA(
		int n_objects,
		float4* d_lower_bound_ptr,
		float4* d_upper_bound_ptr,
		float* d_lowerx_ptr,
		float* d_upperx_ptr,
		float* d_lowery_ptr,
		float* d_uppery_ptr,
		float* d_lowerz_ptr,
		float* d_upperz_ptr)
	{
		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= n_objects) {
			return;
		}

		float4 lower_bound = d_lower_bound_ptr[tid];
		float4 upper_bound = d_upper_bound_ptr[tid];

		d_lowerx_ptr[tid] = lower_bound.x;

		d_lowery_ptr[tid] = lower_bound.y;
		d_lowerz_ptr[tid] = lower_bound.z;

		d_upperx_ptr[tid] = upper_bound.x;
		d_uppery_ptr[tid] = upper_bound.y;
		d_upperz_ptr[tid] = upper_bound.z;

	
	}

	void SAPCollision::projectAABB(
		int n_objects) 
	{

		int threadsPerBlock = 256;
		int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

		projectAABBCUDA << < numBlocks, threadsPerBlock >> > (
			n_objects,
			d_lower_bound_ptr,
			d_upper_bound_ptr,
			d_lower_extent_x_ptr,
			d_upper_extent_x_ptr,
			d_lower_extent_y_ptr,
			d_upper_extent_y_ptr,
			d_lower_extent_z_ptr,
			d_upper_extent_z_ptr
			);
	}





	void SAPCollision::sortLowestExtents(
		int n_objects) 
	{
		SpatialPartition::sortCollisionList(
			(uint32_t*)d_lower_extent_x_ptr,
			(uint32_t*)d_idx_ptr,
			(uint32_t*)d_temp_key_ptr,
			(uint32_t*)d_temp_value_ptr,
			(uint32_t*)d_radix_ptr,
			(uint32_t*)d_radix_sum_ptr,
			n_objects
		);
	}


	__device__ void populateCollisions(
		int tid,
		int& collision_length,
		int* pending_collisions,
		int& idx)
	{

		bool unique_collision = true;
		for (int k = 0; k < 16; k++) { // <-- Max collisions is 10!!
			if (idx == pending_collisions[k]) {

				unique_collision = false;
			}
		}

		if (unique_collision == false) {
			return;
		}
		else {

			if (collision_length <= 8 && idx != tid) {
				pending_collisions[collision_length] = idx;
				collision_length += 1;
			} 

			if (collision_length > 16) {
				printf("More collisions than expected!\n");
			}

			return;
		}
	}

	__device__ void removeDuplicates(int* input_array, int* output_array, int array_size, int collision_size) {

		int output_size = 0;

		for (int i = 0; i < array_size; i++) {
			float currentValue = input_array[i];
			bool is_duplicate = false;

			for (int j = 0; j < output_size; j++) {
				if (currentValue == output_array[j]) {
					is_duplicate = true;
					break;
				}
			}

			if (!is_duplicate && output_size < collision_size) {
				output_array[output_size] = currentValue;
				output_size++;
			}
		}
	}

	__global__ void sweepAndPrunePrimaryCUDA(
		int n_objects,
		float* upperx,
		float* lowerx,
		int* idxx,
		int* potential_collision,
		int padding)
	{

		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= n_objects) {
			return;
		}

		int pending_collisions[10];

		int home_idx = tid;
		int sorted_home_idx = idxx[home_idx];
		float home_upper_extent = upperx[sorted_home_idx];

		int phantom_idx;
		float phantom_lower_extent;

		int pending_collision_length = 0;

		int n_tid = tid + 512;
		if (n_tid >= n_objects) {
			n_tid = n_objects;
		}

		int j = 0;

		for (int i = tid + 1; i < n_tid; i++) {

			if (i == home_idx) {
				continue;
			}

			phantom_lower_extent = lowerx[i];

			phantom_idx = idxx[i];

			// Check primary proj
			if (phantom_lower_extent <= home_upper_extent) {

				populateCollisions(tid, pending_collision_length, potential_collision + tid * 256, phantom_idx);

			}
			//home_upper_extent = uppery[sorted_home_idx];
			//phantom_lower_extent = lowery[phantom_idx];

			//// Check Y proj
			//if (phantom_lower_extent <= home_upper_extent) {

			//	home_upper_extent = upperz[sorted_home_idx];
			//	phantom_lower_extent = lowerz[phantom_idx];

			//	// Check Z proj
			//	if (phantom_lower_extent <= home_upper_extent) {

			//		//coll_counter += 1;
			//		collection[j] = phantom_idx;
					//populateCollisions(tid, pending_collision_length, potential_collision + tid * padding, phantom_idx);
					//printf("Collision detected between: %d and %d\n", sorted_home_idx, phantom_idx);

				//}
			//}
		//}
		//j++;
	//}

	//removeDuplicates(collection, potential_collision + tid * padding, size, padding);
		}
	}

	__global__ void sweepAndPruneCUDA(
		int n_objects,
		float* upperx,
		float* lowerx,
		float* uppery,
		float* lowery,
		int* idxx,
		int* idxy,
		int* potential_collision,
		int padding)
	{

		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= n_objects) {
			return;
		}

		int pending_collisions[10];

		int home_idx = tid;
		int sorted_home_idx = idxx[home_idx];
		float home_upper_extent = upperx[sorted_home_idx];

		int phantom_idx;
		float phantom_lower_extent;

		int pending_collision_length = 0;

		int n_tid = tid + 64;
		if (n_tid >= n_objects) {
			n_tid = n_objects;
		}

		int j = 0;

		for (int i = tid + 1; i < n_tid; i++) {

			if (i == home_idx) {
				continue;
			}

			phantom_lower_extent = lowerx[i];

			phantom_idx = idxx[i];

			// Check primary proj
			if (phantom_lower_extent <= home_upper_extent) {

				//populateCollisions(tid, pending_collision_length, potential_collision + tid * 256, phantom_idx);

				home_upper_extent = uppery[sorted_home_idx];
				phantom_lower_extent = lowery[phantom_idx];

				// Check Y proj
				if (phantom_lower_extent <= home_upper_extent) {

					//coll_counter += 1;
					populateCollisions(tid, pending_collision_length, potential_collision + tid * padding, phantom_idx);
					//printf("Collision detected between: %d and %d\n", sorted_home_idx, phantom_idx);

				}
			}
		}
	}


	__global__ void sweepAndPruneSecondaryCUDA(
		int n_objects,
		float* upper,
		float* lower,
		int* idxx,
		int* potential_collision,
		int padding)
	{
		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= n_objects) {
			return;
		}

		int home_idx = tid;

		float home_upper_extent = upper[home_idx];

		for (int i = 0; i < padding; i++) {

			int phantom_idx = potential_collision[tid * padding + i];

			float phantom_lower_extent = lower[phantom_idx];

			if (phantom_lower_extent > home_upper_extent) {
				potential_collision[tid * padding + i] = -1;
			}
		}
	}


	// Sweep and Prune Block Implementation
	__global__ void sweepAndPruneBlocksCUDA(
		int n_objects,
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
		int padding)
	{

		unsigned int bid = blockIdx.x;

		if (bid >= n_objects) {
			return;
		}

		int* idx;
		float* upper;
		float* lower;
		int* coll;

		int collision = 0;

		int coll_counter = 0;

		int pending_collisions[10];

		int home_idx = bid;
		int sorted_home_idx = idxx[home_idx];
		float home_upper_extent = upperx[sorted_home_idx];

		int phantom_idx;
		float phantom_lower_extent;

		int pending_collision_length = 0;

		int n_tid = bid + padding;
		if (n_tid >= n_objects) {
			n_tid = n_objects;
		}
		else {

		}

		int i = threadIdx.x + blockDim.x * blockIdx.x;

		if (i >= n_objects) {
			return;
		}

		//for (int i = tid + 1; i < n_tid; i++) {

			if (i == home_idx) {
				return;
			}

			phantom_lower_extent = lowerx[i];

			phantom_idx = idxx[i];

			// Check X proj
			if (phantom_lower_extent <= home_upper_extent) { // <-- TODO: change this so it starts with axis with most position variance

				home_upper_extent = uppery[sorted_home_idx];
				phantom_lower_extent = lowery[phantom_idx];

				// Check Y proj
				if (phantom_lower_extent <= home_upper_extent) {

					home_upper_extent = upperz[sorted_home_idx];
					phantom_lower_extent = lowerz[phantom_idx];

					// Check Z proj
					if (phantom_lower_extent <= home_upper_extent) {

						coll_counter += 1;
						//populateCollisions(tid, pending_collision_length, potential_collision + tid * padding, phantom_idx);
						//printf("Collision detected between: %d and %d\n", sorted_home_idx, phantom_idx);

					}
				}
			}
		//}
	}

	void SAPCollision::sweepAndPrune(
		int n_objects) 
	{

		int threadsPerBlock = 256;
		int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

		sweepAndPruneCUDA << < numBlocks, threadsPerBlock >> > (
			n_objects,
			d_upper_extent_x_ptr,
			d_temp_key_ptr,
			d_upper_extent_y_ptr,
			d_lower_extent_y_ptr,
			d_temp_value_ptr,
			d_idx_ptr,
			d_potential_collision_ptr,
			16);

		/*sweepAndPruneSecondaryCUDA << < numBlocks, threadsPerBlock >> > (
			n_objects,
			d_upper_extent_y_ptr,
			d_lower_extent_y_ptr,
			d_idx_ptr,
			d_potential_collision_ptr,
			32);

		sweepAndPruneSecondaryCUDA << < numBlocks, threadsPerBlock >> > (
			n_objects,
			d_upper_extent_z_ptr,
			d_lower_extent_z_ptr,
			d_idx_ptr,
			d_potential_collision_ptr,
			10);*/

		// Only check x, then use that as the key for y and z

		/*int threadsPerBlock = 64;
		int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

		sweepAndPruneBlocksCUDA << < n_objects, threadsPerBlock >> > (
			n_objects,
			d_upper_extent_x_ptr,
			d_temp_key_ptr,
			d_upper_extent_y_ptr,
			d_lower_extent_y_ptr,
			d_upper_extent_z_ptr,
			d_lower_extent_z_ptr,
			d_temp_value_ptr,
			d_idx_ptr,
			d_idx_ptr,
			d_potential_collision_ptr,
			10);*/

	}

	void SAPCollision::processCollisions(
		float4* d_position_ptr,
		float* d_radius_ptr,
		int n_objects,
		int max_collisions
	) 
	{
		SAPCollision::initIdx(
			n_objects,
			max_collisions
		);

		

		SAPCollision::computeAABB(
			d_position_ptr,
			d_radius_ptr,
			n_objects
		);

		

		SAPCollision::projectAABB(
			n_objects
		);

		

		SAPCollision::sortLowestExtents(
			n_objects
		);


		SAPCollision::sweepAndPrune(
			n_objects
		);

		//CUDA_ERR_CHECK(cudaDeviceSynchronize());
	}

}
