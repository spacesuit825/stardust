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

namespace STARDUST {

	///// Device Kernels ///
    __global__ void computeAABBCUDA(
        float4* position,
        float* radius,
        float4* lower,
        float4* upper,
        int n_objects
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

    __global__ void projectAABBCUDA(
        float4* d_lower_bound_ptr,
        float4* d_upper_bound_ptr,
        float* d_lowerx_ptr,
        float* d_upperx_ptr,
        float* d_lowery_ptr,
        float* d_uppery_ptr,
        float* d_lowerz_ptr,
        float* d_upperz_ptr,
        int n_objects)
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

    __global__ void sweepAndPruneCUDA(
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
        int padding)
    {

        unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid >= n_objects) {
            return;
        }

        int* idx;
        float* upper;
        float* lower;
        int* coll;

        int collision = 0;

        int pending_collisions[10];

        int home_idx = tid;
        int sorted_home_idx = idxx[home_idx];
        float home_upper_extent = upperx[sorted_home_idx];

        int phantom_idx;
        float phantom_lower_extent;

        int pending_collision_length = 0;

        int n_tid = tid + 10;
        if (n_tid > n_objects) {
            n_tid = n_objects;
        }
        else {
            n_tid = 10;
        }

        for (int i = tid + 1; i < n_tid; i++) {

            if (i == sorted_home_idx) {
                continue;
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

                        //populateCollisions(tid, pending_collision_length, potential_collision + tid * padding, phantom_idx);
                        //printf("Collision detected between: %d and %d\n", sorted_home_idx, phantom_idx);

                    }
                }
            }
        }

    }



	/// Host Calls ///

	void SpatialPrune::computeAABB(
        float4* position,
        float* radius,
        float4* lower,
        float4* upper,
        int n_objects
	)
	{
        int threadsPerBlock = 256;
        int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

        computeAABBCUDA << < numBlocks, threadsPerBlock >> > (
            position,
            radius,
            lower,
            upper,
            n_objects
            );
	};

	void SpatialPrune::projectAABB(
        float4* d_lower_bound_ptr,
        float4* d_upper_bound_ptr,
        float* d_lowerx_ptr,
        float* d_upperx_ptr,
        float* d_lowery_ptr,
        float* d_uppery_ptr,
        float* d_lowerz_ptr,
        float* d_upperz_ptr,
        int n_objects
	)
	{
        int threadsPerBlock = 256;
        int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

        projectAABBCUDA << < numBlocks, threadsPerBlock >> > (
            d_lower_bound_ptr,
            d_upper_bound_ptr,
            d_lowerx_ptr,
            d_upperx_ptr,
            d_lowery_ptr,
            d_uppery_ptr,
            d_lowerz_ptr,
            d_upperz_ptr,
            n_objects
            );
	};

	void SpatialPrune::sortLowerExtents(
        uint32_t* keys_in,
        uint32_t* values_in,
        uint32_t* keys_out,
        uint32_t* values_out,
        uint32_t* radices,
        uint32_t* radix_sums,
        int n
	)
	{
        // Access the radix sorter in spatial partition. TODO: Modularise this
        SpatialPartition::sortCollisionList(
            keys_in,
            values_in,
            keys_out,
            values_out,
            radices,
            radix_sums,
            n
        );

	};

	void SpatialPrune::sweepAndPrune(
        float* d_upperx_ptr,
        float* d_sorted_lowerx_ptr,
        float* d_uppery_ptr,
        float* d_lowery_ptr,
        float* d_upperz_ptr,
        float* d_lowerz_ptr,
        int* d_sorted_idxx_ptr,
        int* d_idxy_ptr,
        int* d_idxz_ptr,
        int* d_potential_collision_ptr,
        int n_objects,
        int max_collisions
	)
	{
        int threadsPerBlock = 256;
        int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

        sweepAndPruneCUDA << <numBlocks, threadsPerBlock >> > (
            d_upperx_ptr,
            d_sorted_lowerx_ptr,
            d_uppery_ptr,
            d_lowery_ptr,
            d_upperz_ptr,
            d_lowerz_ptr,
            d_sorted_idxx_ptr,
            d_idxy_ptr,
            d_idxz_ptr,
            d_potential_collision_ptr,
            n_objects,
            max_collisions);
	};

}