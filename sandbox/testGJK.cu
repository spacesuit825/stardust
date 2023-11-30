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
#include "cuda_utils.hpp"
#include "helper_math.hpp"
#include "util.hpp"
#include "primitives.hpp"

// Narrow phase receives a collision array with int4s from broad phase
// pair.x = d_idx_ptr[query_node_idx];
// pair.y = d_idx_ptr[rigid_idx];
// pair.z = NEW_PAIR_MARKER;
// pair.w = NEW_PAIR_MARKER;

// sphere-sphere
// triangle-triangle
// sphere-triangle

#define GJK_MAX_ITERATIONS 64 // Arbitrary number, this is reasonably low

namespace STARDUST {

	typedef struct CollisionData {
		bool have_we_collided = false;
		float4 host_position;
		float4 phantom_position;

		float penetration_depth;
	} CollisionData;

	__device__ void supportSphere(
		float4& direction,
		float4& position,
		float radius,
		float4& extent) {

		if (length(direction) == 0.0f) {
			extent = position;
			return;
		}

		extent = position + normalize(direction) * radius;
	}

	__device__ void supportPointSoupNaive(
		float4& direction,
		int vertex_start,
		int vertex_length,
		float4* d_vertex_ptr,
		float4& extent) {

		// Use this for polyhedra and raw triangles
		// O(n) time complexity, but for shapes with low vertex counts, ~ <100 this should be good enough for now

		float test_distance;

		float max_distance = dot(direction, d_vertex_ptr[vertex_start]);
		unsigned int idx = 0;

		for (int i = 1; i < vertex_length; i++) {
			test_distance = dot(direction, d_vertex_ptr[vertex_start + i]);

			if (test_distance > max_distance) {
				max_distance = test_distance;

				idx = i;
			}
		}

		extent = d_vertex_ptr[vertex_start + idx];
	}

	__device__ float4 computeMinkowskiDifference(
		float4* d_vertex_ptr,
		Hull& host_hull,
		Hull& phantom_hull,
		float4& direction
	)
	{
		float4 host_extent;
		float4 phantom_extent;

		// Replace these with a better solution in the future
		switch (host_hull.type) {
			case SPHERE:
				supportSphere(-direction, host_hull.position, host_hull.radius, host_extent);
				break;

			case TRIANGLE:
				supportPointSoupNaive(-direction, host_hull.vertex_idx, host_hull.n_vertices, d_vertex_ptr, host_extent);
				break;

			case POLYHEDRA:
				supportPointSoupNaive(-direction, host_hull.vertex_idx, host_hull.n_vertices, d_vertex_ptr, host_extent);
				break;

		}

		switch (phantom_hull.type) {
			case SPHERE:
				supportSphere(direction, phantom_hull.position, phantom_hull.radius, phantom_extent);
				break;

			case TRIANGLE:
				supportPointSoupNaive(direction, phantom_hull.vertex_idx, phantom_hull.vertex_idx, d_vertex_ptr, phantom_extent);
				break;

			case POLYHEDRA:
				supportPointSoupNaive(direction, phantom_hull.vertex_idx, phantom_hull.n_vertices, d_vertex_ptr, phantom_extent);
				break;

		}

		return phantom_extent - host_extent;

	}

	__device__ void expandSimplex3(
		float4& A,
		float4& B,
		float4& C,
		float4& D,
		int& simplex_dimension,
		float4& search_direction
	)
	{
		bool has_search_updated = false;

		float4 n = make_float4(cross(make_float3(B - A), make_float3(C - A)));
		float4 to_origin = -A;

		simplex_dimension = 2;

		if (dot(make_float4(cross(make_float3(B - A), make_float3(n))), to_origin) > 0 && !(has_search_updated)) {
			C = A;

			search_direction = make_float4(cross(cross(make_float3(B - A), make_float3(to_origin)), make_float3(B - A)));
			has_search_updated = true;
		}

		if (dot(make_float4(cross(make_float3(n), make_float3(C - A))), to_origin) > 0 && !(has_search_updated)) {
			B = A;

			search_direction = make_float4(cross(cross(make_float3(C - A), make_float3(to_origin)), make_float3(C - A)));
			has_search_updated = true;
		}

		if (!(has_search_updated)) simplex_dimension = 3;

		if (dot(n, to_origin) > 0 && !(has_search_updated)) {
			D = C;
			C = B;
			B = A;

			search_direction = n;
			has_search_updated = true;
		}
		else if (!(has_search_updated)) {
			D = B;
			B = A;

			search_direction = -n;
			has_search_updated = true;
		}
	}

	__device__ bool expandSimplex4(
		float4& A,
		float4& B,
		float4& C,
		float4& D,
		int& simplex_dimension,
		float4& search_direction
	)
	{
		bool have_we_collided = false;

		float4 ABC = make_float4(cross(make_float3(B - A), make_float3(C - A)));
		float4 ACD = make_float4(cross(make_float3(C - A), make_float3(D - A)));
		float4 ADB = make_float4(cross(make_float3(D - A), make_float3(B - A)));

		float4 to_origin = -A;

		simplex_dimension = 3;

		if (dot(ABC, to_origin) > 0) {
			D = C;
			C = B;
			B = A;

			search_direction = ABC;
			have_we_collided = false;
		}
		else if (dot(ACD, to_origin) > 0) {
			B = A;

			search_direction = ACD;
			have_we_collided = false;
		}
		else if (dot(ADB, to_origin) > 0) {
			C = D;
			D = B;
			B = A;

			search_direction = ADB;
			have_we_collided = false;
		}
		else {
			have_we_collided = true;
		}

		return have_we_collided;
	}



	__global__ void gjkCUDA(
		unsigned int n_collisions,
		unsigned int n_primitives,
		int4* d_potential_collision_ptr,
		Hull* d_hull_ptr,
		float4* d_vertex_ptr
		//CollisionData* d_collision_data_ptr
	)
	{

		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= n_collisions) {
			return;
		}

		bool have_we_collided; // well, have we?

		CollisionData collision_data;

		float4 A, B, C, D; // define the simplex vertices

		int4 collision = d_potential_collision_ptr[tid];

		// Extract possibly colliding hulls
		Hull host_hull = d_hull_ptr[collision.x];
		Hull phantom_hull = d_hull_ptr[collision.y];

		// Arbitrary search direction
		float4 search_direction =host_hull.position - phantom_hull.position;
		
		C = computeMinkowskiDifference(d_vertex_ptr, host_hull, phantom_hull, search_direction);

		search_direction = -C;

		B = computeMinkowskiDifference(d_vertex_ptr, host_hull, phantom_hull, search_direction);

		if (dot(B, search_direction) < 0.0f) {
			have_we_collided = false;
			return;
		}

		search_direction = make_float4(cross(cross((make_float3(C) - make_float3(B)), -make_float3(B)), make_float3(C) - make_float3(B)));

		if (length(search_direction) == 0.0f) {
			search_direction = make_float4(cross(make_float3(C) - make_float3(B), make_float3(1.0f, 0.0f, 0.0f)));

			if (length(search_direction) == 0.0f) {
				search_direction = make_float4(cross(make_float3(C) - make_float3(B), make_float3(0.0f, 0.0f, -1.0f)));

			}
		}

		int simplex_dimension = 2;

		for (int i = 0; i < GJK_MAX_ITERATIONS; i++) {
			
			A = computeMinkowskiDifference(d_vertex_ptr, host_hull, phantom_hull, search_direction);

			if (dot(A, search_direction) < 0.0f) {
				have_we_collided = false;
				return;
			}

			simplex_dimension++;

			if (simplex_dimension == 3) {
				expandSimplex3(A, B, C, D, simplex_dimension, search_direction);
			}
			else if (expandSimplex4(A, B, C, D, simplex_dimension, search_direction)) {
				have_we_collided = true;
			}
		}
		
		if (have_we_collided)
			printf("Yo, you just collided!\n");
	}
	
}

void main() {
	unsigned int n_primitives = 2;
	unsigned int n_vertices = 4;
	unsigned int n_collisions = 1;

	thrust::host_vector<STARDUST::Hull> hulls(n_primitives);
	thrust::host_vector<float4> vertex(n_vertices);
	thrust::host_vector<int4> potential_collision(n_collisions);

	float4 point1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 point2 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
	float4 point3 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
	float4 point4 = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

	vertex[0] = point1;
	vertex[1] = point2;
	vertex[2] = point3;
	vertex[3] = point4;

	hulls[0].type = 2;
	hulls[0].position = (1 / 4) * (point1 + point2 + point3 + point4); // Barycentre/centroid
	hulls[0].radius = -1.0f;
	hulls[0].vertex_idx = 0;
	hulls[0].n_vertices = 4;


	hulls[1].type = 0;
	hulls[1].position = make_float4(0.0f, 0.0f, 1.5f, 0.0f);
	hulls[1].radius = 0.5f;

	potential_collision[0].x = 0;
	potential_collision[0].y = 1;
	potential_collision[0].z = -1;
	potential_collision[0].w = -1;

	thrust::device_vector<STARDUST::Hull> d_hulls = hulls;
	thrust::device_vector<float4> d_vertex = vertex;
	thrust::device_vector<int4> d_potential_collision = potential_collision;

	STARDUST::Hull* d_hull_ptr = thrust::raw_pointer_cast(d_hulls.data());
	float4* d_vertex_ptr = thrust::raw_pointer_cast(d_vertex.data());
	int4* d_potential_collision_ptr = thrust::raw_pointer_cast(d_potential_collision.data());

	int threads_per_block = 256;
	int num_blocks = (n_collisions + threads_per_block - 1) / threads_per_block;

	STARDUST::gjkCUDA << < num_blocks, threads_per_block >> > (
		n_collisions,
		n_primitives,
		d_potential_collision_ptr,
		d_hull_ptr,
		d_vertex_ptr
		);

	cudaDeviceSynchronize();
}