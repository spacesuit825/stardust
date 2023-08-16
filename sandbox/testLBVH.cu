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

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

// CUB
#include <cub/device/device_radix_sort.cuh>

// Internal
#include <../src/engine/cuda/cuda_utils.hpp>
#include <../src/engine/cuda/cuda_helpers.cuh>
#include "util.hpp"


#define INVALID_COMMON_PREFIX 128
#define ROOT_NODE_MARKER -1
#define TRAVERSAL_STACK_SIZE 128
#define NEW_PAIR_MARKER -1

// Testing a Linear Bounding Volume Hierarchy for large-scale broad-phase analysis

struct ComputeAABB
{
	__host__ __device__
		AABB operator()(const float4& position, float radius) const
	{
		return AABB(position, radius);
	}
};

void computeAABB(
	int n_objects,
	float4* d_position_ptr,
	float* d_radius_ptr,
	AABB* d_aabb_ptr
)
{
	thrust::transform(thrust::device, d_position_ptr, d_position_ptr + n_objects, d_radius_ptr, d_aabb_ptr, ComputeAABB());
}

struct constructSubDomain
{
	inline __host__ __device__
		AABB operator()(const AABB& a, const AABB& b) const
	{
		AABB subdomain;

		// Compute subdomain
		subdomain.lower_extent.x = min(a.lower_extent.x, b.lower_extent.x);
		subdomain.upper_extent.x = max(a.upper_extent.x, b.upper_extent.x);
		subdomain.lower_extent.y = min(a.lower_extent.y, b.lower_extent.y);
		subdomain.upper_extent.y = max(a.upper_extent.y, b.upper_extent.y);
		subdomain.lower_extent.z = min(a.lower_extent.z, b.lower_extent.z);
		subdomain.upper_extent.z = max(a.upper_extent.z, b.upper_extent.z);

		return subdomain;
	}
};

AABB constructGlobalDomain(
	int n_objects,
	AABB* d_aabb_ptr
)
{
	AABB global_domain;

	return thrust::reduce(thrust::device, d_aabb_ptr, d_aabb_ptr + n_objects, global_domain, constructSubDomain());
}

// Morton Code Pipeline
__device__ unsigned int interleaveBits(
	unsigned int x
)
{
	x &= 0x000003FF;

	x = (x ^ (x << 16)) & 0xFF0000FF;
	x = (x ^ (x << 8)) & 0x0300F00F;
	x = (x ^ (x << 4)) & 0x030C30C3;
	x = (x ^ (x << 2)) & 0x09249249;

	return x;
}

__device__ unsigned int computeMortonCode(
	int x,
	int y,
	int z
)
{
	return interleaveBits(x) << 0 | interleaveBits(y) << 1 | interleaveBits(z) << 2;
}

__global__ void computeSpaceFillingCurveCUDA(
	int n_objects,
	AABB global_domain,
	AABB* d_aabb_ptr,
	unsigned int* d_z_code_ptr,
	int* d_idx_ptr
)
{

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int leaf_node_idx = tid;

	if (leaf_node_idx >= n_objects) {
		return;
	}

	float4 domain_center = (global_domain.lower_extent + global_domain.upper_extent) * 0.5f;
	float4 domain_cell_size = (global_domain.upper_extent - global_domain.lower_extent) / (float)1024;

	AABB aabb = d_aabb_ptr[leaf_node_idx];
	float4 aabb_centroid = (aabb.lower_extent + aabb.upper_extent) * 0.5f;
	float4 aabb_centroid_relative_to_domain = aabb_centroid - domain_center;

	float4 domain_position = aabb_centroid_relative_to_domain / domain_cell_size;

	int4 discrete_position;
	discrete_position.x = (int)((domain_position.x >= 0.0f) ? domain_position.x : floor(domain_position.x));
	discrete_position.y = (int)((domain_position.y >= 0.0f) ? domain_position.y : floor(domain_position.y));
	discrete_position.z = (int)((domain_position.z >= 0.0f) ? domain_position.z : floor(domain_position.z));

	discrete_position = max(make_int4(-512, -512, -512, 0), min(discrete_position, make_int4(511, 511, 511, 0)));
	discrete_position += make_int4(512, 512, 512, 0);

	unsigned int morton_code = computeMortonCode(discrete_position.x, discrete_position.y, discrete_position.z);

	d_z_code_ptr[leaf_node_idx] = morton_code;
	d_idx_ptr[leaf_node_idx] = leaf_node_idx;
}

void computeSpaceFillingCurve(
	int n_objects,
	AABB& global_domain,
	AABB* d_aabb_ptr,
	unsigned int* d_z_code_ptr,
	int* d_idx_ptr
)
{
	int threads_per_block = 256;
	int num_blocks = (n_objects + threads_per_block  - 1) / threads_per_block;

	computeSpaceFillingCurveCUDA << < num_blocks, threads_per_block >> > (
		n_objects,
		global_domain,
		d_aabb_ptr,
		d_z_code_ptr,
		d_idx_ptr
		);
}

// Compute prefix length pipeline
__device__ int64_t upsample(
	int a, 
	int b
) 
{
	return (((int64_t)a) << 32) | b;
}

__device__ int computeCommonPrefixLength(
	int64_t i, 
	int64_t j
) 
{
	return (int)__clz(i ^ j);
}

__device__ int64_t computeCommonPrefix(
	int64_t i, 
	int64_t j
) 
{
	int64_t common_prefix_length = (int64_t)computeCommonPrefixLength(i, j);

	int64_t shared_bits = i & j;
	int64_t bit_mask = ((int64_t)(~0)) << (64 - common_prefix_length);

	return shared_bits & bit_mask;
}

__device__ int computeSharedPrefixLength(
	int64_t prefix_a,
	int prefix_length_a,
	int64_t prefix_b,
	int prefix_length_b
)
{
	return min(computeCommonPrefixLength(prefix_a, prefix_b), min(prefix_length_a, prefix_length_b));
}

__global__ void generatePrefixesCUDA(
	int n_objects,
	int n_internal_nodes,
	int n_nodes,
	unsigned int* d_z_code_ptr,
	int* d_idx_ptr,
	int64_t* d_common_prefixes_ptr,
	int64_t* d_common_prefix_lengths_ptr
)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int internal_node_idx = tid;

	if (internal_node_idx >= n_internal_nodes) {
		return;
	}

	int left_leaf_index = internal_node_idx;
	int right_leaf_index = internal_node_idx + 1;

	unsigned int left_leaf_code = d_z_code_ptr[left_leaf_index];
	unsigned int right_leaf_code = d_z_code_ptr[right_leaf_index];

	int64_t non_duplicate_left_code = upsample(left_leaf_code, left_leaf_index);
	int64_t non_duplicate_right_code = upsample(right_leaf_code, right_leaf_index);

	d_common_prefixes_ptr[internal_node_idx] = computeCommonPrefix(non_duplicate_left_code, non_duplicate_right_code);
	d_common_prefix_lengths_ptr[internal_node_idx] = computeCommonPrefixLength(non_duplicate_left_code, non_duplicate_right_code);
}

void generatePrefixes(
	int n_objects,
	int n_internal_nodes,
	int n_nodes,
	unsigned int* d_z_code_ptr,
	int* d_idx_ptr,
	int64_t* d_common_prefixes_ptr,
	int64_t* d_common_prefix_lengths_ptr
)
{
	int threads_per_block = 256;
	int num_blocks = (n_internal_nodes + threads_per_block - 1) / threads_per_block;

	generatePrefixesCUDA << < num_blocks, threads_per_block >> > (
		n_objects,
		n_internal_nodes,
		n_nodes,
		d_z_code_ptr,
		d_idx_ptr,
		d_common_prefixes_ptr,
		d_common_prefix_lengths_ptr
		);
}

__device__ int isLeafNode(
	int idx
)
{
	return (idx >> 31 == 0);
}

__device__ int setInternalMarker(
	int is_leaf,
	int idx
)
{
	return (is_leaf) ? idx : (idx | 0x80000000);
}

__device__ int removeInternalMarker(
	int idx
)
{
	return idx & (~0x80000000);
}

__global__ void buildBinaryRadixLeavesCUDA(
	int n_objects,
	int64_t* d_common_prefix_lengths_ptr,
	int* d_leaf_parent_nodes,
	int2* d_internal_child_nodes
)
{
	// Note: n_objects == num leaves
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int leaf_node_idx = tid;

	if (leaf_node_idx >= n_objects) {
		return;
	}

	int n_internal_nodes = n_objects - 1;

	int left_split_idx = leaf_node_idx - 1;
	int right_split_idx = leaf_node_idx;

	int left_common_prefix = (left_split_idx >= 0) ? d_common_prefix_lengths_ptr[left_split_idx] : INVALID_COMMON_PREFIX;
	int right_common_prefix = (right_split_idx < n_internal_nodes) ? d_common_prefix_lengths_ptr[right_split_idx] : INVALID_COMMON_PREFIX;

	int is_left_bigger = (left_common_prefix > right_common_prefix);

	if (left_common_prefix == INVALID_COMMON_PREFIX) is_left_bigger = false;
	if (right_common_prefix == INVALID_COMMON_PREFIX) is_left_bigger = true;

	int parent_node_idx = (is_left_bigger) ? left_split_idx : right_split_idx;
	d_leaf_parent_nodes[leaf_node_idx] = parent_node_idx;

	int is_right_child = (is_left_bigger);

	int is_leaf = 1;

	// Cast the int2 from a struct to a pointer (allows us to index in)
	int* child_node_int = (int*)(&d_internal_child_nodes[parent_node_idx]);
	child_node_int[is_right_child] = setInternalMarker(is_leaf, leaf_node_idx);
}

void buildBinaryRadixLeaves(
	int n_objects,
	int64_t* d_common_prefix_lengths_ptr,
	int* d_leaf_parent_nodes,
	int2* d_internal_child_nodes
)
{
	int threads_per_block = 256;
	int num_blocks = (n_objects + threads_per_block - 1) / threads_per_block;

	buildBinaryRadixLeavesCUDA << < num_blocks, threads_per_block >> > (
		n_objects,
		d_common_prefix_lengths_ptr,
		d_leaf_parent_nodes,
		d_internal_child_nodes
		);
}


__global__ void buildBinaryRadixInternalNodesCUDA(
	int n_internal_nodes,
	int64_t* d_common_prefixes_ptr,
	int64_t* d_common_prefix_lengths_ptr,
	int2* d_internal_child_nodes_ptr,
	int* d_internal_parent_nodes_ptr,
	int* d_root_node_idx_ptr
)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int internal_node_idx = tid;

	if (internal_node_idx >= n_internal_nodes) {
		return;
	}

	int64_t node_prefix = d_common_prefixes_ptr[internal_node_idx];
	int64_t node_prefix_length = d_common_prefix_lengths_ptr[internal_node_idx];

#ifndef BINARY_SEARCH_CONSTRUCTION // DEFAULT TO BINARY SEARCH CONSTRUCTION
	
	if (tid == 10) {
		printf("Defaulting to Binary Search\n");
	}

	int left_idx = -1;
	{
		int lower = 0;
		int upper = internal_node_idx - 1;

		while (lower <= upper) {
			int mid = (lower + upper) / 2;
			int64_t mid_prefix = d_common_prefixes_ptr[mid];
			int mid_prefix_length = d_common_prefix_lengths_ptr[mid];

			int mid_shared_prefix_length = computeSharedPrefixLength(node_prefix, node_prefix_length, mid_prefix, mid_prefix_length);

			if (mid_shared_prefix_length < node_prefix_length) {
				int right = mid + 1;
				
				if (right < internal_node_idx) {
					int64_t right_prefix = d_common_prefixes_ptr[right];
					int right_prefix_length = d_common_prefix_lengths_ptr[right];

					int right_shared_prefix_length = computeSharedPrefixLength(node_prefix, node_prefix_length, right_prefix, right_prefix_length);
					
					if (right_shared_prefix_length < node_prefix_length) {
						lower = right;
						left_idx = right;
					}
					else {
						left_idx = mid;
						break;
					}
				}
				else {
					left_idx = mid;
					break;
				}
			}
			else {
				upper = mid - 1;
			}
		}
	}

	int right_idx = -1;
	{
		int lower = internal_node_idx + 1;
		int upper = n_internal_nodes - 1;

		while (lower <= upper) {
			int mid = (lower + upper) / 2;
			int64_t mid_prefix = d_common_prefixes_ptr[mid];
			int mid_prefix_length = d_common_prefix_lengths_ptr[mid];

			int mid_shared_prefix_length = computeSharedPrefixLength(node_prefix, node_prefix_length, mid_prefix, mid_prefix_length);

			if (mid_shared_prefix_length < node_prefix_length) {
				int left = mid - 1;

				if (left > internal_node_idx) {
					int64_t left_prefix = d_common_prefixes_ptr[left];
					int left_prefix_length = d_common_prefix_lengths_ptr[left];

					int left_shared_prefix_length = computeSharedPrefixLength(node_prefix, node_prefix_length, left_prefix, left_prefix_length);

					if (left_shared_prefix_length < node_prefix_length) {
						upper = left;
						right_idx = left;
					}
					else {
						right_idx = mid;
						break;
					}
				}
				else {
					right_idx = mid;
					break;
				}
			}
			else {
				lower = mid + 1;
			}
		}
	}

#else

	if (tid == 10) {
		printf("Defaulting to Linear Search\n");
	}
	
	int left_idx = -1;
	int right_idx = -1;
	
	for (int i = internal_node_idx - 1; i >= 0; --i) {
		int left_shared_prefix_length = 
			computeSharedPrefixLength(
				node_prefix,
				node_prefix_length,
				d_common_prefixes_ptr[i],
				d_common_prefix_lengths_ptr[i]
			);
	
		if (left_shared_prefix_length < node_prefix_length) {
			left_idx = i;
			break;
		}
	}
	
	for (int i = internal_node_idx + 1; i < n_internal_nodes; ++i) {
		int right_shared_prefix_length =
			computeSharedPrefixLength(
				node_prefix,
				node_prefix_length,
				d_common_prefixes_ptr[i],
				d_common_prefix_lengths_ptr[i]
			);
	
		if (right_shared_prefix_length < node_prefix_length) {
			right_idx = i;
			break;
		}
	}
	
#endif

	// Find and select parent
	{
		int left_prefix_length = (left_idx != -1) ? d_common_prefix_lengths_ptr[left_idx] : INVALID_COMMON_PREFIX;
		int right_prefix_length = (right_idx != -1) ? d_common_prefix_lengths_ptr[right_idx] : INVALID_COMMON_PREFIX;

		int is_left_bigger = (left_prefix_length > right_prefix_length);

		if (left_prefix_length == INVALID_COMMON_PREFIX) is_left_bigger = false;
		else if (right_prefix_length == INVALID_COMMON_PREFIX) is_left_bigger = true;

		int parent_node_idx = (is_left_bigger) ? left_idx : right_idx;

		int is_root_node = (left_idx == -1 && right_idx == -1);

		d_internal_parent_nodes_ptr[internal_node_idx] = (!is_root_node) ? parent_node_idx : ROOT_NODE_MARKER;

		int is_leaf = 0;
		if (!is_root_node) {
			int is_right_child = (is_left_bigger);

			int* child_node_int = (int*)(&d_internal_child_nodes_ptr[parent_node_idx]);
			child_node_int[is_right_child] = setInternalMarker(is_leaf, internal_node_idx);
		}

	}
}

void buildBinaryRadixInternalNodes(
	int n_internal_nodes,
	int64_t* d_common_prefixes_ptr,
	int64_t* d_common_prefix_lengths_ptr,
	int2* d_internal_child_nodes_ptr,
	int* d_internal_parent_nodes_ptr,
	int* d_root_node_idx_ptr
)
{
	int threads_per_block = 256;
	int num_blocks = (n_internal_nodes + threads_per_block - 1) / threads_per_block;

	buildBinaryRadixInternalNodesCUDA << < num_blocks, threads_per_block >> > (
		n_internal_nodes,
		d_common_prefixes_ptr,
		d_common_prefix_lengths_ptr,
		d_internal_child_nodes_ptr,
		d_internal_parent_nodes_ptr,
		d_root_node_idx_ptr
		);
}


__global__ void probeBinaryRadixDepthCUDA(
	int n_internal_nodes,
	int* d_root_node_idx_ptr,
	int* d_internal_parent_nodes_ptr,
	int* d_distance_to_root_ptr,
	int* d_max_distance_to_root_ptr
)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid == 0) {
		d_max_distance_to_root_ptr[tid] = 0;
	}

	int internal_node_idx = tid;
	if (internal_node_idx >= n_internal_nodes) {
		return;
	}

	int distance_to_root = 0;
	{
		int parent_idx = d_internal_parent_nodes_ptr[internal_node_idx];

		while (parent_idx != ROOT_NODE_MARKER) {
			parent_idx = d_internal_parent_nodes_ptr[parent_idx];
			++distance_to_root;
		}
	}
	
	d_distance_to_root_ptr[internal_node_idx] = distance_to_root;

	__shared__ int local_max_distance;
	if (threadIdx.x == 0) local_max_distance = 0;

	__syncthreads();

	atomicMax(&local_max_distance, distance_to_root);

	__syncthreads();

	if (threadIdx.x == 0) atomicMax(d_max_distance_to_root_ptr, local_max_distance);
}


void probeBinaryRadixDepth(
	int n_internal_nodes,
	int* d_root_node_idx_ptr,
	int* d_internal_parent_nodes_ptr,
	int* d_distance_to_root_ptr,
	int* d_max_distance_to_root_ptr
)
{
	
	int threads_per_block = 256;
	int num_blocks = (n_internal_nodes + threads_per_block - 1) / threads_per_block;

	probeBinaryRadixDepthCUDA << < num_blocks, threads_per_block >> > (
		n_internal_nodes,
		d_root_node_idx_ptr,
		d_internal_parent_nodes_ptr,
		d_distance_to_root_ptr,
		d_max_distance_to_root_ptr
		);

}

__device__ float4 getMin(
	float4& x,
	float4& y
)
{
	return make_float4(min(x.x, y.x), min(x.y, y.y), min(x.z, y.z), 0.0f);
}

__device__ float4 getMax(
	float4& x,
	float4& y
)
{
	return make_float4(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), 0.0f);
}

__global__ void buildBinaryRadixAABBsCUDA(
	int n_internal_nodes,
	int max_tree_depth,
	int distance_to_root,
	AABB* d_aabb_ptr,
	AABB* d_internal_aabb_ptr,
	unsigned int* d_z_code_ptr,
	int* d_idx_ptr,
	int* d_distance_to_root_ptr,
	int2* d_internal_child_nodes_ptr
)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

	unsigned int internal_node_idx = tid;
	if (internal_node_idx >= n_internal_nodes) {
		return;
	}

	int distance = d_distance_to_root_ptr[internal_node_idx];

	if (distance == distance_to_root) {
		int left_child_idx = d_internal_child_nodes_ptr[internal_node_idx].x;
		int right_child_idx = d_internal_child_nodes_ptr[internal_node_idx].y;

		int is_left_leaf = isLeafNode(left_child_idx);
		int is_right_leaf = isLeafNode(right_child_idx);

		left_child_idx = removeInternalMarker(left_child_idx);
		right_child_idx = removeInternalMarker(right_child_idx);

		int left_rigid_idx = (is_left_leaf) ? d_idx_ptr[left_child_idx] : -1;
		int right_rigid_idx = (is_right_leaf) ? d_idx_ptr[right_child_idx] : -1;

		AABB left_child_aabb = (is_left_leaf) ? d_aabb_ptr[left_rigid_idx] : d_internal_aabb_ptr[left_child_idx];
		AABB right_child_aabb = (is_right_leaf) ? d_aabb_ptr[right_rigid_idx] : d_internal_aabb_ptr[right_child_idx];
	
		AABB merged_aabb;
		merged_aabb.lower_extent = getMin(left_child_aabb.lower_extent, right_child_aabb.lower_extent);
		merged_aabb.upper_extent = getMax(left_child_aabb.upper_extent, right_child_aabb.upper_extent);
		
		d_internal_aabb_ptr[internal_node_idx] = merged_aabb;
	}
}

void buildBinaryRadixAABBs(
	int n_internal_nodes,
	int max_tree_depth,
	AABB* d_aabb_ptr,
	AABB* d_internal_aabb_ptr,
	unsigned int* d_z_code_ptr,
	int* d_idx_ptr,
	int* d_distance_to_root_ptr,
	int2* d_internal_child_nodes_ptr
)
{
	int threads_per_block = 256;
	int num_blocks = (n_internal_nodes + threads_per_block - 1) / threads_per_block;

	for (int distance_to_root = max_tree_depth; distance_to_root >= 0; --distance_to_root) {
		buildBinaryRadixAABBsCUDA << < num_blocks, threads_per_block >> > (
			n_internal_nodes,
			max_tree_depth,
			distance_to_root,
			d_aabb_ptr,
			d_internal_aabb_ptr,
			d_z_code_ptr,
			d_idx_ptr,
			d_distance_to_root_ptr,
			d_internal_child_nodes_ptr
			);
	}
}

__global__ void findLeafIndexRangeCUDA(
	int n_internal_nodes,
	int2* d_internal_child_nodes_ptr,
	int2* d_internal_leaf_idx_range_ptr
)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

	unsigned int internal_node_idx = tid;
	if (internal_node_idx >= n_internal_nodes) {
		return;
	}

	int n_leaf_nodes = n_internal_nodes + 1;

	int2 child_nodes = d_internal_child_nodes_ptr[internal_node_idx];

	int2 leaf_idx_range;

	// Lowest leaf
	{
		int lowest_idx = child_nodes.x;

		while (!isLeafNode(lowest_idx)) lowest_idx = d_internal_child_nodes_ptr[removeInternalMarker(lowest_idx)].x;

		leaf_idx_range.x = lowest_idx;
	}

	// Highest leaf
	{
		int highest_idx = child_nodes.y;

		while (!isLeafNode(highest_idx)) highest_idx = d_internal_child_nodes_ptr[removeInternalMarker(highest_idx)].y;

		leaf_idx_range.y = highest_idx;
	}

	d_internal_leaf_idx_range_ptr[internal_node_idx] = leaf_idx_range;
}

void findLeafIndexRange(
	int n_internal_nodes,
	int2* d_internal_child_nodes_ptr,
	int2* d_internal_leaf_idx_range_ptr
)
{
	int threads_per_block = 256;
	int num_blocks = (n_internal_nodes + threads_per_block - 1) / threads_per_block;

	findLeafIndexRangeCUDA << < num_blocks, threads_per_block >> > (
		n_internal_nodes,
		d_internal_child_nodes_ptr,
		d_internal_leaf_idx_range_ptr
		);
}

__device__ bool testAABBCollision(const AABB* aabb1, const AABB* aabb2)
{
	bool overlap = true;
	
	overlap = (aabb1->lower_extent.x > aabb2->upper_extent.x || aabb1->upper_extent.x < aabb2->lower_extent.x) ? false : overlap;
	overlap = (aabb1->lower_extent.z > aabb2->upper_extent.z || aabb1->upper_extent.z < aabb2->lower_extent.z) ? false : overlap;
	overlap = (aabb1->lower_extent.y > aabb2->upper_extent.y || aabb1->upper_extent.y < aabb2->lower_extent.y) ? false : overlap;
	
	return overlap;
}

__global__ void findPotentialCollisionsCUDA(
	int n_objects,
	int max_collisions,
	int* d_idx_ptr,
	int* d_n_pairs_ptr,
	int4* d_overlapping_pairs_ptr,
	AABB* d_aabb_ptr,
	int* d_root_node_idx_ptr,
	int2* d_internal_child_nodes_ptr,
	AABB* d_internal_aabb_ptr,
	int2* d_internal_leaf_idx_range_ptr,
	unsigned int* d_z_code_ptr
)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int query_node_idx = tid;
	if (query_node_idx >= n_objects) {
		return;
	}

	int query_idx = d_idx_ptr[query_node_idx];

	AABB query_aabb = d_aabb_ptr[query_idx];

	int stack[TRAVERSAL_STACK_SIZE];

	int stack_size = 1;
	stack[0] = *d_root_node_idx_ptr;

	//printf("root node: %d\n", stack[0]);

	while (stack_size) {

		int internal_leaf_idx = stack[stack_size - 1];

		--stack_size;

		//printf("hello\n");

		int is_leaf = isLeafNode(internal_leaf_idx);
		int node_idx = removeInternalMarker(internal_leaf_idx);

		{
			int highest_leaf_idx = (is_leaf) ? node_idx : d_internal_leaf_idx_range_ptr[node_idx].y;
			
			//printf("leaf contents: %d, %d\n", d_internal_leaf_idx_range_ptr[node_idx].x, d_internal_leaf_idx_range_ptr[node_idx].y);
			
			//printf("highest leaf idx: %d, %d\n", highest_leaf_idx, query_node_idx);
			
			
			// BROKEN MUST FIX
			//if (highest_leaf_idx <= query_node_idx) continue;
		}

		int rigid_idx = (is_leaf) ? d_idx_ptr[node_idx] : -1;

		AABB node_aabb = (is_leaf) ? d_aabb_ptr[rigid_idx] : d_internal_aabb_ptr[node_idx];

		//printf("Query aabb: %.3f, %.3f, %.3f\n", node_aabb.lower_extent.x, node_aabb.lower_extent.y, node_aabb.lower_extent.z);
		//printf("Node aabb: %.3f, %.3f, %.3f\n", query_aabb.lower_extent.x, query_aabb.lower_extent.y, query_aabb.lower_extent.z);
		
		if (testAABBCollision(&query_aabb, &node_aabb)) {
			if (is_leaf) {
				int4 pair;

				pair.x = d_idx_ptr[query_node_idx];
				pair.y = d_idx_ptr[rigid_idx];
				pair.z = NEW_PAIR_MARKER;
				pair.w = NEW_PAIR_MARKER;

				atomicAdd(&d_n_pairs_ptr[0], 1);
				int pair_idx = d_n_pairs_ptr[0];

				//printf("%d\n", pair_idx);

				if (pair_idx < max_collisions) d_overlapping_pairs_ptr[pair_idx] = pair;
			}

			if (!is_leaf) {
				if (stack_size + 2 > TRAVERSAL_STACK_SIZE) {
					// ERROR
				}
				else {
					stack[stack_size++] = d_internal_child_nodes_ptr[node_idx].x;
					stack[stack_size++] = d_internal_child_nodes_ptr[node_idx].y;
				}
			}
		}
	}

}

void findPotentialCollisions(
	int n_objects,
	int max_collisions,
	int* d_idx_ptr,
	int* d_n_pairs_ptr,
	int4* d_overlapping_pairs_ptr,
	AABB* d_aabb_ptr,
	int* d_root_node_idx_ptr,
	int2* d_internal_child_nodes_ptr,
	AABB* d_internal_aabb_ptr,
	int2* d_internal_leaf_idx_range_ptr,
	unsigned int* d_z_code_ptr
)
{
	int threads_per_block = 256;
	int num_blocks = (n_objects + threads_per_block - 1) / threads_per_block;

	findPotentialCollisionsCUDA << < num_blocks, threads_per_block >> > (
		n_objects,
		max_collisions,
		d_idx_ptr,
		d_n_pairs_ptr,
		d_overlapping_pairs_ptr,
		d_aabb_ptr,
		d_root_node_idx_ptr,
		d_internal_child_nodes_ptr,
		d_internal_aabb_ptr,
		d_internal_leaf_idx_range_ptr,
		d_z_code_ptr
		);
}

void sortKeys(
	int n_objects,
	uint32_t* d_z_code_ptr,
	int* d_idx_ptr
)
{
	thrust::sort_by_key(thrust::device, d_z_code_ptr, d_z_code_ptr + n_objects, d_idx_ptr);
}



float randomFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}


int main() {

	int n_objects = 1000000;
	int max_collisions = n_objects * 10;

	int n_internal_nodes = n_objects - 1;
	int n_nodes = 2 * n_objects - 1;

	// Allocate vectors
	thrust::host_vector<float4> position(n_objects);
	thrust::host_vector<float> radius(n_objects);
	thrust::host_vector<int> idx(n_objects);

	thrust::host_vector<int> root_node(1);
	thrust::host_vector<int> n_pairs(1);
	thrust::host_vector<unsigned int> z_code(n_objects);
	thrust::host_vector<int> leaf_parent_nodes(n_objects);
	thrust::host_vector<int2> internal_child_nodes(n_internal_nodes);
	thrust::host_vector<int> internal_parent_nodes(n_internal_nodes);
	thrust::host_vector<int> distance_to_root(n_internal_nodes);
	thrust::host_vector<int> max_distance_to_root(1);

	thrust::host_vector<int2> internal_leaf_idx_range(n_internal_nodes);

	thrust::host_vector<int64_t> common_prefixes(n_internal_nodes);
	thrust::host_vector<int64_t> common_prefix_lengths(n_internal_nodes);

	thrust::host_vector<AABB> aabb(n_objects);
	thrust::host_vector<AABB> internal_aabb(n_internal_nodes);
	thrust::host_vector<Node> nodes(n_nodes);

	thrust::host_vector<int4> overlapping_pairs(max_collisions);


	// Define the initial values
	for (int i = 0; i < n_objects; i++) {
		position[i] = make_float4(randomFloat(0.0, 5.0), randomFloat(0.0, 5.0), randomFloat(0.0, 5.0), 0.0);
		radius[i] = 1.0;
		idx[i] = i;
	}

	n_pairs[0] = 0;

	// Send vectors to the device
	thrust::device_vector<float4> d_position = position;
	thrust::device_vector<float> d_radius = radius;
	thrust::device_vector<int> d_idx = idx;

	thrust::device_vector<int> d_root_node = root_node;
	thrust::device_vector<unsigned int> d_z_code = z_code;
	thrust::device_vector<int> d_leaf_parent_nodes = leaf_parent_nodes;
	thrust::device_vector<int2> d_internal_child_nodes = internal_child_nodes;
	thrust::device_vector<int> d_internal_parent_nodes = internal_parent_nodes;
	thrust::device_vector<int> d_distance_to_root = distance_to_root;
	thrust::device_vector<int> d_max_distance_to_root = max_distance_to_root;
	thrust::device_vector<int64_t> d_common_prefixes = common_prefixes;
	thrust::device_vector<int64_t> d_common_prefix_lengths = common_prefix_lengths;

	thrust::device_vector<int2> d_internal_leaf_idx_range = internal_leaf_idx_range;

	thrust::device_vector<AABB> d_aabb = aabb;
	thrust::device_vector<AABB> d_internal_aabb = internal_aabb;
	thrust::device_vector<Node> d_nodes = nodes;
	thrust::device_vector<int4> d_overlapping_pairs = overlapping_pairs;
	thrust::device_vector<int> d_n_pairs = n_pairs;

	// Extract raw pointers
	float4* d_position_ptr = thrust::raw_pointer_cast(d_position.data());
	float* d_radius_ptr = thrust::raw_pointer_cast(d_radius.data());
	int* d_idx_ptr = thrust::raw_pointer_cast(d_idx.data());

	int* d_root_node_idx_ptr = thrust::raw_pointer_cast(d_root_node.data());
	unsigned int* d_z_code_ptr = thrust::raw_pointer_cast(d_z_code.data());
	int* d_leaf_parent_nodes_ptr = thrust::raw_pointer_cast(d_leaf_parent_nodes.data());
	int2* d_internal_child_nodes_ptr = thrust::raw_pointer_cast(d_internal_child_nodes.data());
	int* d_internal_parent_nodes_ptr = thrust::raw_pointer_cast(d_internal_parent_nodes.data());
	int* d_distance_to_root_ptr = thrust::raw_pointer_cast(d_distance_to_root.data());
	int* d_max_distance_to_root_ptr = thrust::raw_pointer_cast(d_max_distance_to_root.data());
	int64_t* d_common_prefixes_ptr = thrust::raw_pointer_cast(d_common_prefixes.data());
	int64_t* d_common_prefix_lengths_ptr = thrust::raw_pointer_cast(d_common_prefix_lengths.data());

	int2* d_internal_leaf_idx_range_ptr = thrust::raw_pointer_cast(d_internal_leaf_idx_range.data());

	AABB* d_aabb_ptr = thrust::raw_pointer_cast(d_aabb.data());
	AABB* d_internal_aabb_ptr = thrust::raw_pointer_cast(d_internal_aabb.data());
	Node* d_nodes_ptr = thrust::raw_pointer_cast(d_nodes.data());

	int4* d_overlapping_pairs_ptr = thrust::raw_pointer_cast(d_overlapping_pairs.data());

	int* d_n_pairs_ptr = thrust::raw_pointer_cast(d_n_pairs.data());

	// Initiate Nodes
	//thrust::transform(thrust::device, d_nodes_ptr, d_nodes_ptr + n_nodes, d_nodes_ptr, InitNodes());

	printf("Starting LBVH construction...\n");

	// LBVH pipeline
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceSynchronize();
	cudaEventRecord(start);

	// 1.) Compute AABB for all objects in the scene (embarassingly parallel)
	computeAABB(
		n_objects,
		d_position_ptr,
		d_radius_ptr,
		d_aabb_ptr
	);

	// 2.) Use these boxes to compute a tight global domain which we can subdivide (requires thrust reduce)
	AABB global_domain = constructGlobalDomain(
		n_objects,
		d_aabb_ptr
	);

	printf("Global domain is: %.3f %.3f, %.3f\n", global_domain.upper_extent.x, global_domain.upper_extent.y, global_domain.upper_extent.z);

	// 3.) Compute Morton codes for every AABB
	computeSpaceFillingCurve(
		n_objects,
		global_domain,
		d_aabb_ptr,
		d_z_code_ptr,
		d_idx_ptr
	);

	// 4.) Sort the Morton codes (use the idx array so we know how to get back)
	sortKeys(
		n_objects,
		d_z_code_ptr,
		d_idx_ptr
	);

	printf("Preparing to build leaves...\n");

	// 5.) Build Binary Radix Tree
	generatePrefixes(
		n_objects,
		n_internal_nodes,
		n_nodes,
		d_z_code_ptr,
		d_idx_ptr,
		d_common_prefixes_ptr,
		d_common_prefix_lengths_ptr
	);

	buildBinaryRadixLeaves(
		n_objects,
		d_common_prefix_lengths_ptr,
		d_leaf_parent_nodes_ptr,
		d_internal_child_nodes_ptr
	);

	buildBinaryRadixInternalNodes(
		n_internal_nodes,
		d_common_prefixes_ptr,
		d_common_prefix_lengths_ptr,
		d_internal_child_nodes_ptr,
		d_internal_parent_nodes_ptr,
		d_root_node_idx_ptr
	);

	probeBinaryRadixDepth(
		n_internal_nodes,
		d_root_node_idx_ptr,
		d_internal_parent_nodes_ptr,
		d_distance_to_root_ptr,
		d_max_distance_to_root_ptr
	);

	max_distance_to_root = d_max_distance_to_root;
	int max_tree_depth = max_distance_to_root[0];

	buildBinaryRadixAABBs(
		n_internal_nodes,
		max_tree_depth,
		d_aabb_ptr,
		d_internal_aabb_ptr,
		d_z_code_ptr,
		d_idx_ptr,
		d_distance_to_root_ptr,
		d_internal_child_nodes_ptr
	);
	
	//printf("Tree construction complete... %d levels deep\n", max_tree_depth);

	printf("Initiating tree traversal process...\n");

	printf("Leaf index ranges found...\n");
	
	findLeafIndexRange(
		n_internal_nodes,
		d_internal_child_nodes_ptr,
		d_internal_leaf_idx_range_ptr
	);

	findPotentialCollisions(
		n_objects,
		max_collisions,
		d_idx_ptr,
		d_n_pairs_ptr,
		d_overlapping_pairs_ptr,
		d_aabb_ptr,
		d_root_node_idx_ptr,
		d_internal_child_nodes_ptr,
		d_internal_aabb_ptr,
		d_internal_leaf_idx_range_ptr,
		d_z_code_ptr
	);

	// TODO: 
	// 1.) Fix double-testing
	// 2.) Implement collision table

	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	n_pairs = d_n_pairs;
	max_distance_to_root = d_max_distance_to_root;

	printf("Tree construction and traversal completed in %.5f seconds.\n", milliseconds / 1000.0f);
	printf("%d Collisions detected...\n", n_pairs[0]);

	printf("%d level tree\n", max_distance_to_root[0]);

	return 0;
}