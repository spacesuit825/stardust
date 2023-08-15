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

struct ComputeUpperExtent
{
	__host__ __device__
		float4 operator()(const float4& position, float radius) const
	{
		float4 upper = make_float4(position.x + radius, position.y + radius, position.z + radius, 1.0f);
		return upper;
	}
};

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
		subdomain.xmin = min(a.xmin, b.xmin);
		subdomain.xmax = max(a.xmax, b.xmax);
		subdomain.ymin = min(a.ymin, b.ymin);
		subdomain.ymax = max(a.ymax, b.ymax);
		subdomain.zmin = max(a.zmin, b.zmin);
		subdomain.zmax = max(a.zmax, b.zmax);
		
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





int main() {
	
	int n_objects = 100000;
	int max_depth = 4;

	// Allocate vectors
	thrust::host_vector<float4> position(n_objects);
	thrust::host_vector<float> radius(n_objects);
	thrust::host_vector<int> idx(n_objects);
	thrust::host_vector<int> tags(n_objects);
	thrust::host_vector<AABB> aabb(n_objects);

	// Define the initial values
	for (int i = 0; i < n_objects; i++) {
		position[i] = make_float4(1.0, 0.0, 0.0, 0.0);
		radius[i] = 0.5;
		idx[i] = i;
	}

	// Send vectors to the device
	thrust::device_vector<float4> d_position = position;
	thrust::device_vector<float> d_radius = radius;
	thrust::device_vector<int> d_idx = idx;
	thrust::device_vector<int> d_tags = tags;
	thrust::device_vector<AABB> d_aabb = aabb;

	// Extract raw pointers
	float4* d_position_ptr = thrust::raw_pointer_cast(d_position.data());
	float* d_radius_ptr = thrust::raw_pointer_cast(d_radius.data());
	int* d_idx_ptr = thrust::raw_pointer_cast(d_idx.data());
	int* d_tags_ptr = thrust::raw_pointer_cast(d_tags.data());
	AABB* d_aabb_ptr = thrust::raw_pointer_cast(d_aabb.data());

	// Sweep and prune pipeline

	// Create an octree to hold all our points (we can SAP each leaf of the octree for optimal performance)
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

	// 3.) Build 




	aabb = d_aabb;
	tags = d_tags;

	printf("global domain max: %.3f, %.3f, %.3f", global_domain.xmax, global_domain.ymax, global_domain.zmax);

	if (n_objects < 10) {
		printf("[");
		for (int i = 0; i < n_objects; i++) {
			printf("%d, ", tags[i]);//aabb[i].xmin);
		}
		printf("]\n");
	}

	return 0;
}