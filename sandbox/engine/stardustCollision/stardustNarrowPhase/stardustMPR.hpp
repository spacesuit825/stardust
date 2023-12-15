#ifndef _STARDUST_MPR_HEADER_
#define _STARDUST_MPR_HEADER_

// C++
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

// CUDA
#include <vector_types.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvfunctional>
#include <cuda_runtime_api.h>
#include <cstdint>
//#include <device_functions.h>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Internal
#include "../../stardustGeometry/stardustPrimitives.hpp"
//#include "../../stardustUtility/cuda_utils.hpp"
#include "../../stardustUtility/helper_math.hpp"
#include "../../stardustUtility/util.hpp"

namespace STARDUST {

	

	class MPR {

	public:

		MPR() {

		};

		~MPR() {

		};

		void allocate(unsigned int, unsigned int, unsigned int);
		void execute(unsigned int, unsigned int, int, const int4*, const Hull*, const float4*);
		//void resetMPR();
		//void destroyMPR();

		CollisionManifold* getCollisionPtr() { return d_collision_manifold_ptr; };
		int getCollisionNumber() { return n_collisions; };

	private:
		void minkowskiPortalRefinement(unsigned int, unsigned int, const int4*, const Hull*, const float4*, int*);

		int n_collisions = 0;

		thrust::host_vector<CollisionManifold> collision_manifold;
		thrust::host_vector<int> n_pairs;

		thrust::device_vector<CollisionManifold> d_collision_manifold;
		thrust::device_vector<int> d_n_pairs;

		CollisionManifold* d_collision_manifold_ptr;
		int* d_n_pairs_ptr;

		int n_collided = 0;

	};
}





#endif // _STARDUST_MPR_HEADER_