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
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

// CUB
#include <cub/device/device_radix_sort.cuh>

// Internal
//#include "../../stardustUtility/cuda_utils.hpp"
#include "../../stardustUtility/helper_math.hpp"
#include "../../stardustUtility/util.hpp"
#include "../../stardustGeometry/stardustPrimitives.hpp"
#include "../../stardustCollision/stardustNarrowPhase/stardustMPR.hpp"

// #include "lean-vtk/include/lean_vtk.hpp"

// Narrow phase receives a collision array with int4s from broad phase
// pair.x = d_idx_ptr[query_node_idx];
// pair.y = d_idx_ptr[rigid_idx];
// pair.z = NEW_PAIR_MARKER;
// pair.w = NEW_PAIR_MARKER;

// sphere-sphere
// triangle-triangle
// sphere-triangle

#define GJK_MAX_ITERATIONS 64 // Arbitrary number, this is reasonably low
#define MPR_MAX_ITERATIONS 64

#define MPR_TOL 1e-4

namespace STARDUST {

	

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
		const float4* d_vertex_ptr,
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

	__device__ float4 getSupportPoint(
		const float4* d_vertex_ptr,
		Hull& hull,
		float4& direction
	)
	{
		float4 extent;

		switch (hull.type) {
		case SPHERE:
			supportSphere(direction, hull.position, hull.radius, extent);
			break;

		case TRIANGLE:
			supportPointSoupNaive(direction, hull.vertex_idx, hull.n_vertices, d_vertex_ptr, extent);
			break;

		case POLYHEDRA:
			supportPointSoupNaive(direction, hull.vertex_idx, hull.n_vertices, d_vertex_ptr, extent);
			break;
		}

		return extent;
	}


	__device__ float4 computeMinkowskiDifference(
		const float4* d_vertex_ptr,
		Hull& host_hull,
		Hull& phantom_hull,
		float4& direction
	)
	{

		/*switch (host_hull.type) {
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
				supportPointSoupNaive(direction, phantom_hull.vertex_idx, phantom_hull.n_vertices, d_vertex_ptr, phantom_extent);
				break;

			case POLYHEDRA:
				supportPointSoupNaive(direction, phantom_hull.vertex_idx, phantom_hull.n_vertices, d_vertex_ptr, phantom_extent);
				break;

		}*/

		return getSupportPoint(d_vertex_ptr, phantom_hull, direction) - getSupportPoint(d_vertex_ptr, host_hull, -direction);

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

		float4 A, B, C, D; // define the simplex vertices

		int4 collision = d_potential_collision_ptr[tid];

		// Extract possibly colliding hulls
		Hull host_hull = d_hull_ptr[collision.x];
		Hull phantom_hull = d_hull_ptr[collision.y];

		// Arbitrary search direction
		float4 search_direction = host_hull.position - phantom_hull.position;

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

		if (have_we_collided) {
			printf("Yo, you just collided!\n");
			//collision_data.have_we_collided = have_we_collided;
		}
	}

	__device__ void swap(float4& a, float4& b) {

		float4 tmp = a;

		a = b;
		b = tmp;
	}

	__device__ bool zeroPhaseMPR(
		Hull& host_hull,
		Hull& phantom_hull,
		const float4* d_vertex_ptr,
		float& penetration,
		float4& normal,
		float4& point_A,
		float4& point_B
	)
	{

		float4 v, a, a1, a2, b, b1, b2, c, c1, c2, d, d1, d2; // define the simplex vertices

		float4 search_direction;

		v = phantom_hull.position - host_hull.position;

		if (length(v) == 0.0f) {
			v = make_float4(1e-5, 0.0f, 0.0f, 0.0f);
		}

		search_direction = -v;

		a1 = getSupportPoint(d_vertex_ptr, host_hull, -search_direction);
		a2 = getSupportPoint(d_vertex_ptr, phantom_hull, search_direction);

		a = a2 - a1;

		if (dot(a, search_direction) <= MPR_TOL) {
			// COLLISION FAILURE
			return false;
		}


		search_direction = make_float4(cross(make_float3(a), make_float3(v)));
		if (length(search_direction) == 0.0f) {
			//COLLISION SUCCESS

			normal = normalize(a - v);

			point_A = a1;
			point_B = a2;

			float4 position = 0.5 * (a1 + a2);

			penetration = abs(dot((a1 - position), -normal)) + abs(dot((a2 - position), normal));
			return true;
		}

		b1 = getSupportPoint(d_vertex_ptr, host_hull, -search_direction);
		b2 = getSupportPoint(d_vertex_ptr, phantom_hull, search_direction);

		b = b2 - b1;

		if (dot(b, search_direction) == 0.0f) {
			// COLLISION FAILURE
			return false;
		}


		search_direction = make_float4(cross(make_float3(a - v), make_float3(b - v)));
		float distance_to_origin = dot(search_direction, v);

		if (distance_to_origin > 0.0f) {
			swap(a, b);
			swap(a1, b1);
			swap(a2, b2);

			search_direction = -search_direction;
		}

		float triple_product;

		for (int i = 0; i < MPR_MAX_ITERATIONS; i++) {

			c1 = getSupportPoint(d_vertex_ptr, host_hull, -search_direction);
			c2 = getSupportPoint(d_vertex_ptr, phantom_hull, search_direction);

			c = c2 - c1;

			if (dot(c, search_direction) == 0.0f) {
				// COLLISION FAILURE
				return false;
			}

			triple_product = dot(make_float4(cross(make_float3(a), make_float3(c))), v);
			if (triple_product < 0.0f) {
				b = c;
				b1 = c1;
				b2 = c2;

				search_direction = make_float4(cross(make_float3(a - v), make_float3(c - v)));
				continue;
			}

			triple_product = dot(make_float4(cross(make_float3(c), make_float3(b))), v);
			if (triple_product < 0.0f) {
				a = c;
				a1 = c1;
				a2 = c2;

				search_direction = make_float4(cross(make_float3(c - v), make_float3(b - v)));
				continue;
			}

			bool hit = false;
			int phase_2 = 0;

			for (int j = 0; j < MPR_MAX_ITERATIONS; j++) {

				phase_2++;

				search_direction = make_float4(cross(make_float3(b - a), make_float3(c - a)));

				if (length(search_direction) == 0.0f) {
					// COLLISION SUCCESS
					return false;
				}

				search_direction = normalize(search_direction);

				float distance = dot(search_direction, a);

				if (distance >= 0.0f && !hit) {
					// COLLISION SUCCESS, HIT DETECTED

					normal = search_direction;

					float h0 = dot(make_float4(cross(make_float3(a), make_float3(b))), c);
					float h1 = dot(make_float4(cross(make_float3(c), make_float3(b))), v);
					float h2 = dot(make_float4(cross(make_float3(v), make_float3(a))), c);
					float h3 = dot(make_float4(cross(make_float3(b), make_float3(a))), v);

					float sum = h0 + h1 + h2 + h3;

					if (sum <= 0.0f) {

						h0 = 0.0f;
						h1 = dot(make_float4(cross(make_float3(b), make_float3(c))), search_direction);
						h2 = dot(make_float4(cross(make_float3(c), make_float3(a))), search_direction);
						h3 = dot(make_float4(cross(make_float3(a), make_float3(b))), search_direction);

						sum = h1 + h2 + h3;
					}

					float inverse = 1.0f / sum;

					float4 pA = (h0 * phantom_hull.position + h1 * a1 + h2 * b1 + h3 * c1) * inverse;
					float4 pB = (h0 * host_hull.position + h1 * a2 + h2 * b2 + h3 * c2) * inverse;

					float4 suppA = getSupportPoint(d_vertex_ptr, host_hull, -search_direction);
					float4 suppB = getSupportPoint(d_vertex_ptr, phantom_hull, search_direction);

					float4 pointA = pA + abs(dot((suppA - pA), -search_direction));
					float4 pointB = pB + abs(dot((suppB - pB), search_direction));

					float4 position = 0.5 * (pointA + pointB);

					point_A = pointA;
					point_B = pointB;

					penetration = abs(dot((pointA - position), -normal)) + abs(dot((pointB - position), -normal));
					hit = true;
				}


				d1 = getSupportPoint(d_vertex_ptr, host_hull, -search_direction);
				d2 = getSupportPoint(d_vertex_ptr, phantom_hull, search_direction);

				d = d2 - d1;

				float delta = dot((d - c), search_direction);
				float separation = -dot(d, search_direction);

				if (delta <= MPR_TOL || separation >= 0.0f) {
					// COLLISION SUCCESS
					normal = search_direction;
					return hit; //hit;
				}


				float dividing_face_1 = dot(make_float4(cross(make_float3(d), make_float3(a))), v);
				float dividing_face_2 = dot(make_float4(cross(make_float3(d), make_float3(b))), v);
				float dividing_face_3 = dot(make_float4(cross(make_float3(d), make_float3(c))), v);

				if (dividing_face_1 < 0.0f) {
					if (dividing_face_2 < 0.0f) {
						a = d;
						a1 = d1;
						a2 = d2;
					}
					else {
						c = d;
						c1 = d1;
						c2 = d2;
					}
				}
				else {
					if (dividing_face_3 < 0.0f) {
						b = d;
						b1 = d1;
						b2 = d2;
					}
					else {
						a = d;
						a1 = d1;
						a2 = d2;
					}
				}
			}
		}

	}

	__global__ void minkowskiPortalRefinementCUDA(
		unsigned int n_collisions,
		unsigned int n_primitives,
		CollisionManifold* d_collision_manifold_ptr,
		const int4* d_potential_collision_ptr,
		const Hull* __restrict__ d_hull_ptr,
		const float4* __restrict__ d_vertex_ptr,
		int* d_n_pairs_ptr
	)
	{
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= n_collisions) {
			return;
		}

		bool have_we_collided; // well, have we?

		CollisionManifold collision_data;

		int4 collision = d_potential_collision_ptr[tid];

		//printf("\n MPR host: %d, phan: %d\n", collision.x, collision.y);

		// Extract possibly colliding hulls
		Hull host_hull = d_hull_ptr[collision.x];
		Hull phantom_hull = d_hull_ptr[collision.y];

		

		float penetration = 0.0f;
		float4 normal;
		float4 pointA;
		float4 pointB;

		//printf("Collision1\n");

		have_we_collided = zeroPhaseMPR(
			host_hull,
			phantom_hull,
			d_vertex_ptr,
			penetration,
			normal,
			pointA,
			pointB);

		if (have_we_collided) {

			// Build collision manifold

			collision_data.collision_normal = normal;
			collision_data.pointA = pointA;
			collision_data.pointB = pointB;
			collision_data.penetration_depth = penetration;

			collision_data.host_hull_idx = collision.x;
			collision_data.phantom_hull_idx = collision.y;

			int pair_idx = atomicAdd(&d_n_pairs_ptr[0], 1);

			// Add something to stop writing over the end of the array!!!

			//printf("Penetration depth: %.3f\n", collision_data.penetration_depth);
			
			d_collision_manifold_ptr[pair_idx] = collision_data;
			
		}

	}

	void MPR::minkowskiPortalRefinement(
		unsigned int n_collisions,
		unsigned int n_primitives,
		const int4* d_potential_collision_ptr,
		const Hull* d_hull_ptr,
		const float4* d_vertex_ptr,
		int* d_n_pairs_ptr
	)
	{
		int threads_per_block = 256;
		int num_blocks = (n_collisions + threads_per_block - 1) / threads_per_block;

		minkowskiPortalRefinementCUDA << < num_blocks, threads_per_block >> > (
			n_collisions,
			n_primitives,
			d_collision_manifold_ptr,
			d_potential_collision_ptr,
			d_hull_ptr,
			d_vertex_ptr,
			d_n_pairs_ptr
			);
	}

	void MPR::allocate(
		unsigned int n_collisions,
		unsigned int n_primitives,
		unsigned int max_collisions
	)
	{

		collision_manifold.resize(max_collisions);
		n_pairs.resize(1);

		d_collision_manifold = collision_manifold;
		d_n_pairs = n_pairs;

		d_collision_manifold_ptr = thrust::raw_pointer_cast(d_collision_manifold.data());
		d_n_pairs_ptr = thrust::raw_pointer_cast(d_n_pairs.data());
	}

	void MPR::reset() {
		d_n_pairs[0] = 0;
	}

	void MPR::execute(
		unsigned int n_collisions,
		unsigned int n_primitives,
		int max_collisions,
		const int4* d_potential_collision_ptr,
		const Hull* d_hull_ptr,
		const float4* d_vertex_ptr
	)
	{
		minkowskiPortalRefinement(
			n_collisions,
			n_primitives,
			d_potential_collision_ptr,
			d_hull_ptr,
			d_vertex_ptr,
			d_n_pairs_ptr
		);

		n_pairs = d_n_pairs;
		n_collided = n_pairs[0];
	}


}