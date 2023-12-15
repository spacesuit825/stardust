#ifndef _STARDUST_LBVH_HEADER_
#define _STARDUST_LBVH_HEADER_

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

	
	
	class LBVH {
	
	public:

		LBVH() {

		}

		~LBVH() {
			
		}

		void allocate(int, int); // Allocates required support arrays for the LBVH tree
		void execute(int n_primitives, int max_collisions, Hull* d_hull_ptr, AABB* d_aabb_ptr); // 
		//void resetLBVH();
		//void destroyLBVH();


		int4* getPotentialCollisionPtr() { return d_overlapping_pairs_ptr; };
		int getCollisionNumber() { return n_collisions; };

	private:
		void computeAABB(int, Hull*, AABB*);
		AABB constructGlobalDomain(int, AABB*);
		void computeSpaceFillingCurve(int, AABB&, AABB*, unsigned int*, int*);
		void generatePrefixes(int, int, int, unsigned int*, int*, int64_t*, int64_t*);
		void buildBinaryRadixLeaves(int, int64_t*, int*, int2*);
		void buildBinaryRadixInternalNodes(int, int64_t*, int64_t*, int2*, int*, int*);
		void probeBinaryRadixDepth(int, int*, int*, int*, int*);
		void buildBinaryRadixAABBs(int, int, AABB*, AABB*, unsigned int*, int*, int*, int2*);
		void findLeafIndexRange(int, int2*, int2*);
		void findPotentialCollisions(int, int, Hull*, int*, int*, int*, int4*, AABB*, int*, int2*, AABB*, int2*, unsigned int*);
		void sortKeys(int, uint32_t*, int*);

		int n_collisions = 0;

		thrust::host_vector<int> idx;
		thrust::host_vector<int> o_type;
		thrust::host_vector<int> root_node;
		thrust::host_vector<int> n_pairs;
		thrust::host_vector<unsigned int> z_code;
		thrust::host_vector<int> leaf_parent_nodes;
		thrust::host_vector<int2> internal_child_nodes;
		thrust::host_vector<int> internal_parent_nodes;
		thrust::host_vector<int> distance_to_root;
		thrust::host_vector<int> max_distance_to_root; //
		thrust::host_vector<int2> internal_leaf_idx_range;
		thrust::host_vector<int64_t> common_prefixes;
		thrust::host_vector<int64_t> common_prefix_lengths;
		//thrust::host_vector<AABB> aabb;
		thrust::host_vector<AABB> internal_aabb;
		thrust::host_vector<int4> overlapping_pairs;
		thrust::host_vector<int4> sorted_pairs;

		thrust::device_vector<int> d_type;
		thrust::device_vector<int> d_idx;
		thrust::device_vector<int> d_root_node;
		thrust::device_vector<unsigned int> d_z_code;
		thrust::device_vector<int> d_leaf_parent_nodes;
		thrust::device_vector<int2> d_internal_child_nodes;
		thrust::device_vector<int> d_internal_parent_nodes;
		thrust::device_vector<int> d_distance_to_root;
		thrust::device_vector<int> d_max_distance_to_root;
		thrust::device_vector<int64_t> d_common_prefixes;
		thrust::device_vector<int64_t> d_common_prefix_lengths;
		thrust::device_vector<int2> d_internal_leaf_idx_range;
		//thrust::device_vector<AABB> d_aabb;
		thrust::device_vector<AABB> d_internal_aabb;
		thrust::device_vector<int4> d_overlapping_pairs;
		thrust::device_vector<int4> d_sorted_pairs;
		thrust::device_vector<int> d_n_pairs;

		int* d_type_ptr;
		int* d_idx_ptr;
		int* d_root_node_idx_ptr;
		unsigned int* d_z_code_ptr;
		int* d_leaf_parent_nodes_ptr;
		int2* d_internal_child_nodes_ptr;
		int* d_internal_parent_nodes_ptr;
		int* d_distance_to_root_ptr;
		int* d_max_distance_to_root_ptr;
		int64_t* d_common_prefixes_ptr;
		int64_t* d_common_prefix_lengths_ptr;
		int2* d_internal_leaf_idx_range_ptr;
		//AABB* d_aabb_ptr;
		AABB* d_internal_aabb_ptr;
		int4* d_overlapping_pairs_ptr;
		int4* d_sorted_pairs_ptr;
		int* d_n_pairs_ptr;
	};


}




#endif // _STARDUST_LBVH_HEADER_