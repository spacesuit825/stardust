#ifndef _STARDUST_SAP_COLLISION_DETECTION_HEADER_
#define _STARDUST_SAP_COLLISION_DETECTION_HEADER_

// Internal
#include "../../cuda_utils.hpp"

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

	class SAPCollision {
		
	public:

		SAPCollision() {
			initiatePointers();
		}

		void initiatePointers();
		void prepareData();
		void allocateCUDA(int, int);
		void transferDataToDevice();
		void processCollisions(float4*, float*, int);

		void initIdx(int);
		void computeAABB(float4*, float*, int);
		void projectAABB(int);
		void sortLowestExtents(int);
		void sweepAndPrune(int);

		void reactCollisions();

		// Pointers
		int* d_idx_ptr;

		float4* d_lower_bound_ptr;
		float4* d_upper_bound_ptr;

		float* d_lower_extent_x_ptr;
		float* d_upper_extent_x_ptr;

		float* d_lower_extent_y_ptr;
		float* d_upper_extent_y_ptr;

		float* d_lower_extent_z_ptr;
		float* d_upper_extent_z_ptr;

		int* d_potential_collision_ptr;

		float* d_temp_key_ptr;
		int* d_temp_value_ptr;

		int* d_radix_ptr;
		int* d_radix_sum_ptr;


	private:

	};

}


#endif // _STARDUST_SAP_COLLISION_DETECTION_HEADER_